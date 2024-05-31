import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
import numpy as np

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

# returns a class
def get_class(kls):
    parts = kls.split('.')
    # exclude the last one
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    # enumerate里面的代表的是被分成多批，一批n_pixels的0-total_pixels的索引。每一批单独构建数据，再append到split里面
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'depth' in data:
            data['depth'] = torch.index_select(model_input['depth'], 1, indx)
        if 'instance_mask' in data:
            data['instance_mask'] = torch.index_select(model_input['instance_mask'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1: #一维张量
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_time():
    torch.cuda.synchronize()
    return time.time()

trans_topil = transforms.ToPILImage()


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        # 创建一个网格，meshgrid包含两个array，第一个代表网格中的点的横坐标，第二个...纵坐标
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2*height*width
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # 生成一个全1的矩阵

        # stack:展平id_coords得到一个(2, height*width)的张量
        # unsqueeze:扩展得到(1, 2, height*width)
        # repeat:得到(batch_size, 2, height*width)
        # cat:得到(batch_size, 3, height*width) 齐次化
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords) #矩阵相乘，得到相机坐标下的像素坐标
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points #缩放深度
        cam_points = torch.cat([cam_points, self.ones], 1) #齐次化
        return cam_points
