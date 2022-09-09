import random
import os
import os.path as osp
from PIL import Image
import numpy as np

root_dir = './datasets/training_img'
save_path_inf = './datasets/cropped_img/Inf/'
save_path_vis = './datasets/cropped_img/Vis/'
dataset_inf = os.listdir(os.path.join(root_dir, 'Inf'))
dataset_vis = os.listdir(os.path.join(root_dir, 'Vis'))


for idx in dataset_inf:
    orignal = Image.open(os.path.join(root_dir, 'Inf', idx))
    orignal_size = orignal.size
    gt = Image.open(os.path.join(root_dir, 'Vis', idx)).resize((orignal_size))
    setp = 32
    x = 0
    y = 0
    w = 128
    h = 128

    while y * setp + h < orignal_size[1]:
        while x * setp + w < orignal_size[0]:
            orignal_region = orignal.crop((x * setp, y * setp, x * setp + w, y * setp + h))
            orignal_region.save(save_path_inf + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=x * setp, haha=y * setp))
            gt_region = gt.crop((x * setp, y * setp, x * setp + w, y * setp + h))
            gt_region.save(save_path_vis + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=x * setp, haha=y * setp))
            x = x + 1
        orignal_region = orignal.crop((orignal.size[0] - setp, y * setp, orignal.size[0] - setp + w, y * setp + h))
        orignal_region.save(save_path_inf + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=orignal.size[0] - setp, haha=y * setp))
        gt_region = gt.crop((orignal.size[0] - setp, y * setp, orignal.size[0] - setp + w, y * setp + h))
        gt_region.save(save_path_vis + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=orignal.size[0] - setp, haha=y * setp))
        x = 0
        y = y + 1
    while x * setp + w < orignal_size[0]:
        orignal_region = orignal.crop((x * setp, orignal.size[1] - setp, x * setp + w, orignal.size[1] - setp + h))
        orignal_region.save(save_path_inf + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=x * setp, haha=orignal.size[1] - setp))
        gt_region = gt.crop((x * setp, orignal.size[1] - setp, x * setp + w, orignal.size[1] - setp + h))
        gt_region.save(save_path_vis + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=x * setp, haha=orignal.size[1] - setp))
        x = x + 1
    orignal_region = orignal.crop(
        (orignal.size[0] - setp, orignal.size[1] - setp, orignal.size[0] - setp + w, orignal.size[1] - setp + h))
    orignal_region.save(save_path_inf + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=orignal.size[0] - setp, haha=orignal.size[1] - setp))
    gt_region = gt.crop(
        (orignal.size[0] - setp, orignal.size[1] - setp, orignal.size[0] - setp + w, orignal.size[1] - setp + h))
    gt_region.save(save_path_vis + idx.split('.')[0] + '_{loop}_{haha}.jpg'.format(loop=orignal.size[0] - setp, haha=orignal.size[1] - setp))
