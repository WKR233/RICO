import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import trimesh

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth

import torch.distributed as dist

class RICOTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        # conf is a <class 'pyhocon.config_tree.ConfigTree'>
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.infer_stage = kwargs['infer_stage']

        # conf.get_string('train.expname') == str(conf['train']['expname'])
        self.expname = self.conf.get_string('train.expname') + kwargs['expname'] 
        # conf.get_int('dataset.scan_id') == int(conf['dataset']['scan_id'])
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            # fill in {0} with the 0th parameter: '_scan_id'
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            # if the path exists, we need to continue
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                # listdir returns a list, includes the files in the path
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                # path exists, but no timestamp
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    # now the timestamp is the biggest(latest)
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            # create timestamp folder
            # if the parameter do not exist, mkdir
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            # copy conf to exps/rico_scannet/timestamp/runconf.conf
            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        # import the module in datasets, and pass the dataset(a dict) as parameters
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=50000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        # use total iterations to compute how many epochs
        # self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))
        
        if len(self.train_dataset.label_mapping) > 0:
            # a hack way to let network know how many categories, so don't need to manually set in config file
            self.conf['model']['implicit_network']['d_out'] = len(self.train_dataset.label_mapping)
            print('RUNNING FOR {0} CLASSES'.format(len(self.train_dataset.label_mapping)))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        instance_ids = self.train_dataset.instance_ids
        print('Instance IDs: ', instance_ids)
        print('Label mappings: ', self.train_dataset.label_mapping)

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        
        # current model uses MLP and a unified lr
        print('using optimizer w unified lr')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-15)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        # after decay_steps steps, lr *= decay_rate
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=False)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()
        self.n_sem = self.conf.get_int('model.implicit_network.d_out')
        assert self.n_sem == len(self.train_dataset.label_mapping)

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):

        if self.infer_stage:

            # for fast inference, select some important views
            importance_views_list = [
                36, 38, 56, 62, 102, 108, 109, 117, 121, 129, 143, 
                160, 162, 171, 185, 193, 209, 224, 227, 232, 239,
                244, 259, 264, 301, 302, 303
            ]

            plot_mesh = True        # export mesh for the first view

            # NOTE: temp code for obj_bbox_dict
            mesh_root_path = '../exps/RICO_scannet_1/2024_05_08_10_59_45/plots'
            mesh_epoch = 700
            obj_bbox_dict = {}
            obj_num = self.model.module.implicit_network.d_out
            for mesh_idx in range(obj_num):
                mesh_path = os.path.join(mesh_root_path, f'surface_{mesh_epoch}_{mesh_idx}.ply')
                mesh = trimesh.load(mesh_path)
                x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
                y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
                z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
                
                obj_bbox_dict[mesh_idx] = [[x_min, y_min, z_min], [x_max, y_max, z_max]]

            # inference stage, render all views
            print('********** inference stage, render all views **********')
            print('CKPTS: ', self.checkpoints_path)
            for data_index, (indices, model_input, ground_truth) in enumerate(tqdm(self.train_dataloader)):

                if indices[0] not in importance_views_list:
                    continue

                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s, indices)
                    d = {'rgb_values': out['rgb_values'].detach(),
                        'normal_map': out['normal_map'].detach(),
                        'depth_values': out['depth_values'].detach()}
                    if 'rgb_un_values' in out:
                        d['rgb_un_values'] = out['rgb_un_values'].detach()
                    if 'semantic_values' in out:
                        d['semantic_values'] = torch.argmax(out['semantic_values'].detach(),dim=1)
                    res.append(d)
                
                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'])

                save_finish_path = os.path.join(self.plots_dir, 'finish_rendering')
                os.makedirs(save_finish_path, exist_ok=True)

                plt.plot(
                    self.model.module.implicit_network,
                    indices,
                    plot_data,
                    save_finish_path,
                    self.start_epoch,
                    self.img_res,
                    **self.plot_conf,
                    plot_mesh=plot_mesh,
                    obj_bbox_dict=obj_bbox_dict
                )

                # only export mesh for the first view
                plot_mesh = False

            print('SAVE PATH: ', save_finish_path)
            # print finish inference stage
            print('********** finish inference stage **********')

            return 0

        print("training...")
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):
            # 需要保存：保存checkpoint
            if (self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0) or (self.GPU_INDEX == 0 and epoch == self.nepochs):
                self.save_checkpoints(epoch)
            # 需要可视化（可视化==1&&epoch是可视化频率的倍数）
            if (self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0) or (self.GPU_INDEX == 0 and self.do_vis and epoch == self.nepochs):
                self.model.eval() #设置eval模式

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                #拆分输入数据以适应模型->utils/general.py
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s, indices)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach(),
                         'depth_values': out['depth_values'].detach(),
                         'semantic_values': out['semantic_values'].detach()}
                    res.append(d) #存储训练结果

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size) #合并splited的输出->utils/general.py
                plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['instance_mask'])
                print(epoch)
                print(plot_data)
                plot_mesh = True
                #准备好所有数据之后可以画图
                plt.plot_rico(
                    self.model.module.implicit_network,
                    indices,
                    plot_data,
                    self.plots_dir,
                    epoch,
                    self.img_res,
                    plot_mesh,
                    **self.plot_conf
                )

                self.model.train()
            self.train_dataset.change_sampling_idx(self.num_pixels)
            #对于训练数据
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                model_input['instance_mask'] = ground_truth["instance_mask"].cuda().reshape(-1).long()
                
                self.optimizer.zero_grad()
                #计算模型输出
                model_outputs = self.model(model_input, indices, iter_step=self.iter_step)
                #计算loss
                loss_output = self.loss(model_outputs, ground_truth, iter_ratio=self.iter_step / self.max_total_iters)
                loss = loss_output['loss']
                loss.backward() #反向传播loss
                self.optimizer.step() #更新优化器参数
                # Peak Signal-to-Noise Ratio 峰值信噪比
                # reshape(-1,3)代表第一维度未知，需要根据元素数量和后面的参数计算
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                
                self.iter_step += 1                
                # 把数据写入writer
                if self.GPU_INDEX == 0:
                    if data_index % 25 == 0:
                        head_str = '{0}_{1} [{2}] ({3}/{4}): '.format(self.expname, self.timestamp, epoch, data_index, self.n_batches)
                        loss_print_str = ''
                        for k, v in loss_output.items():
                            loss_print_str = loss_print_str + '{0} = {1}, '.format(k, v.item())
                        print_str = head_str + loss_print_str + 'psnr = {0}'.format(psnr.item())
                        print(print_str)
                    
                    for k, v in loss_output.items():
                        self.writer.add_scalar(f'Loss/{k}', v.item(), self.iter_step)
                    
                    self.writer.add_scalar('Statistics/s_value', self.model.module.get_s_value().item(), self.iter_step)
                    self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, semantic_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        #归一到[0, 1]
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        #移动到同一个设备上
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        # save point cloud
        # 0和1是图像的高度和宽度
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)

        # semantic map
        # 最大值对应最有可能的类别
        semantic_map = model_outputs['semantic_values'].argmax(dim=-1).reshape(batch_size, num_samples, 1)
        # in label mapping, 0 is bg idx and 0
        # for instance, first fg is 3 and 1
        # so when using argmax, the output will be label_mapping idx if correct
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
            "semantic_map": semantic_map,
            "semantic_gt": semantic_gt,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        # 相机内参矩阵的逆
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0) #backproject->utils/general.py，输入深度图+内参，得到点位置
        points = torch.cat([points, color], dim=-1) #将位置与颜色拼接起来
        return points.detach().cpu().numpy()
