import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class BUSI(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        if mode == 'Training':
            mode = 'train'
        if mode == 'Test':
            mode = 'test'

        self.name_list = os.listdir(os.path.join(data_path,mode,'images'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1

        # Modify depends on the structure of the Dataset folder

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        msk_path = os.path.join(self.data_path, self.mode, 'labels', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training': # Training Dataset
        #     img_path = os.path.join(self.data_path, 'train', 'images')
        #     msk_path = os.path.join(self.data_path, 'train/labels')
        # else: #Test Dataset
        #     img_path = os.path.join(self.data_path, 'test/images')
        # name = self.name_list[index]
        # img_path = os.path.join(self.data_path, name)
        
        # mask_name = self.label_list[index]
        # msk_path = os.path.join(msk_path, mask_name)

        # img = Image.open(img_path).convert('RGB')
        # mask = Image.open(msk_path).convert('L')

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }