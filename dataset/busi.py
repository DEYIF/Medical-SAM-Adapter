import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click, rect_box


class BUSI(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'train',prompt = 'click', prompt_source = 'gt', plane = False):

        self.name_list = os.listdir(os.path.join(data_path,mode,'prompts'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt_source = prompt_source

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1

        # Modify depends on the structure of the Dataset folder

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        gt_path = os.path.join(self.data_path, self.mode, 'labels', name)
        mask_path = os.path.join(self.data_path, self.mode, 'prompts', name)

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        newsize = (self.img_size, self.img_size)
        gt = gt.resize(newsize)

        if self.prompt == 'click' and self.prompt_source == 'gt':
            point_label, pt = random_click(np.array(gt) / 255, point_label)

        if self.prompt == 'click' and self.prompt_source == 'unet': # prompts provided by U-Net
            point_label, pt = random_click(np.array(mask) / 255, point_label)
        
        if self.prompt == 'box' and self.prompt_source == 'gt':
            pt = rect_box(np.array(gt) / 255)

        if self.prompt == 'box' and self.prompt_source == 'unet': # prompts provided by U-Net
            pt = rect_box(np.array(mask) / 255)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                gt = self.transform_msk(gt).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': gt,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }