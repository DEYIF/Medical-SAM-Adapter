import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click, rect_box, central_click


class BUSI(Dataset):
    # def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'test',prompt = 'click', prompt_source = 'gt', plane = False):

    #     self.prompt_source = prompt_source
    #     if self.prompt_source == 'gt':
    #         self.name_list = os.listdir(os.path.join(data_path,mode,'images'))
    #     elif self.prompt_source == 'unet':
    #         self.name_list = os.listdir(os.path.join(data_path,mode,'prompts'))
    #     self.data_path = data_path
    #     self.mode = mode
    #     self.prompt = prompt
    #     self.img_size = args.image_size
    #     self.transform = transform
    #     self.transform_msk = transform_msk
    
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='test', prompt='click', prompt_source='unet', plane=False):
        self.prompt_source = prompt_source
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk

        self.image_dict = {os.path.splitext(f)[0]: os.path.join(data_path, mode, 'images', f)
                            for f in os.listdir(os.path.join(data_path, mode, 'images'))}
        self.label_dict = {os.path.splitext(f)[0]: os.path.join(data_path, mode, 'labels', f)
                            for f in os.listdir(os.path.join(data_path, mode, 'labels'))}
        self.prompt_dict = {os.path.splitext(f)[0]: os.path.join(data_path, mode, 'prompts', f)
                            for f in os.listdir(os.path.join(data_path, mode, 'prompts'))}

        # 获取交集的键，确保索引时不会出错
        self.name_list = list(self.image_dict.keys() & self.label_dict.keys())  # 只保留都有的文件

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1

        # Modify depends on the structure of the Dataset folder

        """Get the images"""
        name = self.name_list[index]

        img_path = self.image_dict[name]
        gt_path = self.label_dict[name]
        prompt_path = self.prompt_dict.get(name)

        # img_path = os.path.join(self.data_path, self.mode, 'images', name)
        # gt_path = os.path.join(self.data_path, self.mode, 'labels', name)
        # if self.prompt_source == 'gt':
        #     prompt_path = gt_path
        # elif self.prompt_source == 'unet':
        #     prompt_path = os.path.join(self.data_path, self.mode, 'prompts', name)

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        prompt_image = Image.open(prompt_path).convert('L')

        newsize = (self.img_size, self.img_size)
        gt = gt.resize(newsize)
        prompt_image = prompt_image.resize(newsize)

        if self.prompt == 'click':
            # point_label, pt = random_click(np.array(prompt_image) / 255, point_label) 
            # 这里改成不是随机选点，而是选中心的点
            point_label, pt = central_click(np.array(prompt_image) / 255, point_label)
            box_cup = np.array([0,0,0,0])
        elif self.prompt == 'box':
            pt = np.array([0,0])
            prompt_image_np = np.array(prompt_image)
            box_cup = rect_box(prompt_image_np)
        else:
            pt =np.array([0,0])
            box_cup = np.array([0,0,0,0])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                gt = self.transform_msk(gt).int()
                prompt_image = self.transform_msk(prompt_image).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': gt,
            'pt_image':prompt_image,
            'p_label':point_label,
            'pt':pt,
            'box':box_cup,
            'image_meta_dict':image_meta_dict,
        }