from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import json
import random
import imageio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..utils.augmentation import SimpleAugment as RandomGeom
from ..utils.augmentation import RandomIntensity


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.simple_aug = RandomGeom()
        self.intensity_aug = RandomIntensity()
        
        self.crop_from_origin = [cfg.MODEL.img_size] * 3

        self.folder_name = os.path.join(cfg.DATA.data_folder)
        self.folder_name_mam = os.path.join(cfg.DATA.mam_folder)

        group_path = cfg.TRAIN.group_path
        
        if group_path is None:
            self.groups = {'0': os.listdir(self.folder_name)}
        else:
            with open(group_path, 'r') as f:
                self.groups = json.load(f)

    def __getitem__(self, index):

        group_idx = random.randint(0, len(self.groups) - 1)
        current_group = self.groups[str(group_idx)]
        sample_name = random.choice(current_group)

        sample_name = os.path.join(self.folder_name, sample_name)
        sample_att_path = os.path.join(self.folder_name_mam, sample_name)
        
        used_data = imageio.volread(sample_name)
        used_data_att = imageio.volread(sample_att_path)

        raw_data_shape = used_data.shape

        random_z = random.randint(0, raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, raw_data_shape[2]-self.crop_from_origin[2])
                    
        imgs = used_data[random_z:random_z + self.crop_from_origin[0], \
                    random_y:random_y + self.crop_from_origin[1], \
                    random_x:random_x + self.crop_from_origin[2]].copy()
        
        atts = used_data_att[random_z:random_z + self.crop_from_origin[0], \
            random_y:random_y + self.crop_from_origin[1], \
            random_x:random_x + self.crop_from_origin[2]].copy()
        
        [imgs, atts] = self.simple_aug([imgs, atts])

        imgs = self.intensity_aug(imgs)

        imgs = self.scaler(imgs)
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        atts = self.scaler(atts)
        atts = atts[np.newaxis, ...]
        atts = np.ascontiguousarray(atts, dtype=np.float32)

        return imgs, atts

    def __len__(self):
        return int(sys.maxsize)
    
    def scaler(self, img):
        return np.float32(img) / 255.0

class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch


