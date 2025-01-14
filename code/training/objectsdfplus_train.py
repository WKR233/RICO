import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import json
import wandb
import trimesh

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth

class ObjectSDFPlusTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.description = kwargs['description']
        self.use_wandb = kwargs['use_wandb']
        self.infer_stage = kwargs['infer_stage']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        self.finetune_folder = kwargs['ft_folder'] if kwargs['ft_folder'] is not None else None
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            # self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            self.timestamp = f'{self.description}' + '_{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
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

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('[INFO]: shell command : {0}'.format(' '.join(sys.argv)))

        print('[INFO]: Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        # self.all_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)
        # if hasattr(self.all_dataset, 'i_split'): 
        #     # if you would like to split the dataset into train and test, assign 'i_split' attribute to the all_dataset
        #     self.train_dataset = torch.utils.data.Subset(self.all_dataset, self.all_dataset.i_split[0])
        #     self.test_dataset = torch.utils.data.Subset(self.all_dataset, self.all_dataset.i_split[1])
        # else:
        #     self.train_dataset = torch.utils.data.Subset(self.all_dataset, range(self.all_dataset.n_images))
        #     self.test_dataset = torch.utils.data.Subset(self.all_dataset, range(self.all_dataset.n_images))

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('[INFO]: Finish loading data. Data-set size: {0}'.format(self.ds_len))

        if len(self.train_dataset.label_mapping) > 0:
            # a hack way to let network know how many categories, so don't need to manually set in config file
            self.conf['model']['implicit_network']['d_out'] = len(self.train_dataset.label_mapping)
            print('RUNNING FOR {0} CLASSES'.format(len(self.train_dataset.label_mapping)))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8,
                                                            pin_memory=True)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # The MLP and hash grid should have different learning rates
        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        
        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')

            print('[INFO]: Loading pretrained model from {}'.format(old_checkpnts_dir))
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
        
        self.add_objectvio_iter = self.conf.get_int('train.add_objectvio_iter', default=0)

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
            mesh_root_path = '/home/nijunfeng/mycode/project/obj-recon/priorecon/exps/objectsdfplus_mlp_un_scannetpp_1/3n_normal_2024_05_23_20_50_20/plots'
            mesh_epoch = 240
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

            if self.use_wandb:
                infos = json.loads(json.dumps(self.conf))
                wandb.init(
                    config=infos,
                    project=self.conf.get_string('wandb.project_name'),
                    name=self.timestamp,
                    # notes='description',
                    # group='group1 --> tag',
                )

                # # visiualize gradient
                # wandb.watch(self.model, self.optimizer)

        self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
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

                plt.plot(self.model.module.implicit_network,
                        indices,
                        plot_data,
                        self.plots_dir,
                        epoch,
                        self.img_res,
                        **self.plot_conf
                        )

                self.model.train()
            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                self.optimizer.zero_grad()
                
                model_outputs = self.model(model_input, indices, iter_step=self.iter_step)
                model_outputs['epoch'] = epoch
                
                loss_output = self.loss(model_outputs, ground_truth, call_reg=True) if\
                        self.iter_step >= self.add_objectvio_iter else self.loss(model_outputs, ground_truth, call_reg=False)
                # if change the pixel sampling pattern to patch, then you can add a TV loss to enforce some smoothness constraint
                loss = loss_output['loss']
                loss.backward()

                # calculate gradient norm
                total_norm = 0
                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                self.optimizer.step()
                
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                
                self.iter_step += 1                
                
                if self.GPU_INDEX == 0 and data_index %20 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9}, alpha={10}, semantic_loss = {11}, reg_loss = {12}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item(),
                                    loss_output['semantic_loss'].item(),
                                    loss_output['collision_reg_loss'].item()))
                    
                    if self.use_wandb:
                        for k, v in loss_output.items():
                            wandb.log({f'Loss/{k}': v.item()}, self.iter_step)

                        wandb.log({'Statistics/beta': self.model.module.density.get_beta().item()}, self.iter_step)
                        wandb.log({'Statistics/alpha': 1. / self.model.module.density.get_beta().item()}, self.iter_step)
                        wandb.log({'Statistics/psnr': psnr.item()}, self.iter_step)
                        wandb.log({'Statistics/total_norm': total_norm}, self.iter_step)
                        
                        if self.Grid_MLP:
                            wandb.log({'Statistics/lr0': self.optimizer.param_groups[0]['lr']}, self.iter_step)
                            wandb.log({'Statistics/lr1': self.optimizer.param_groups[1]['lr']}, self.iter_step)
                            wandb.log({'Statistics/lr2': self.optimizer.param_groups[2]['lr']}, self.iter_step)
                
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch)

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, seg_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        seg_map = model_outputs['semantic_values'].reshape(batch_size, num_samples)
        seg_gt = seg_gt.to(seg_map.device)

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'seg_gt': seg_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'seg_map': seg_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()
