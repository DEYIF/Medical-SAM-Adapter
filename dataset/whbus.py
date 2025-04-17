import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click, rect_box, central_click, central_box


class WHBUS(Dataset):
    
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='test', prompt='click', prompt_source='unet', plane=False):
        self.prompt_source = prompt_source
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk

        # self.image_dict = {os.path.splitext(f)[0]: os.path.join(data_path, mode, 'images', f)
        #                     for f in os.listdir(os.path.join(data_path, mode, 'images'))}
        
        self.image_dict = {}
        images_folder = os.path.join(data_path, mode, 'images')
        for root, _, files in os.walk(images_folder):
            for file in files:
                # 根据需要判断文件扩展名，这里只加载 jpg/png/jpeg 格式的图片
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # 为防止不同子文件夹中有相同的文件名，可以使用相对路径作为 key
                    rel_path = os.path.relpath(os.path.join(root, file), images_folder)
                    key = os.path.splitext(rel_path)[0]
                    self.image_dict[key] = os.path.join(root, file)

        self.label_dict = {}
        labels_folder = os.path.join(data_path, mode, 'labels')
        for root, _, files in os.walk(labels_folder):
            for file in files:
                # 根据需要判断文件扩展名，这里只加载 jpg/png/jpeg 格式的图片
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # 为防止不同子文件夹中有相同的文件名，可以使用相对路径作为 key
                    rel_path = os.path.relpath(os.path.join(root, file), labels_folder)
                    key = os.path.splitext(rel_path)[0]
                    self.image_dict[key] = os.path.join(root, file)

        self.label_dict = {os.path.splitext(f)[0]: os.path.join(data_path, mode, 'labels', f)
                            for f in os.listdir(os.path.join(data_path, mode, 'labels'))}
        if self.prompt == 'random_click' or self.prompt == 'central_box':
            self.prompt_dict = self.image_dict
        else:
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

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        prompt_image = Image.open(prompt_path).convert('L')

        newsize = (self.img_size, self.img_size)
        gt = gt.resize(newsize)
        prompt_image = prompt_image.resize(newsize)

        if self.prompt == 'click':
            # point_label, pt = random_click(np.array(prompt_image) / 255, point_label) 
            point_label, pt = central_click(np.array(prompt_image) / 255, point_label)
            box_cup = np.array([0,0,0,0])
        elif self.prompt == 'box':
            pt = np.array([0,0])
            prompt_image_np = np.array(prompt_image)
            box_cup = rect_box(prompt_image_np)
        elif self.prompt == 'random_click':
            # Randomly select a point from the image
            point_label, pt = random_click(np.array(prompt_image) / 255, point_label) 
            box_cup = np.array([0,0,0,0])
        elif self.prompt == 'central_box':
            pt = np.array([0,0])
            prompt_image_np = np.array(prompt_image)
            box_cup = central_box(prompt_image_np)
        
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