# video dataset
# we have many folders, each folder contains a video
# each video has a set of images
# this dataset don't have labels, we need to generate labels for it

import os
import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click, rect_box, central_click, central_box


class MICCAI(Dataset):
    
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='test', prompt='click', prompt_source='unet', plane=False):
        self.prompt_source = prompt_source
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        
        # 处理新的数据集结构: 主文件夹包含malignant和benign两个子文件夹，每个子文件夹中有多个视频文件夹
        self.image_paths = []
        self.video_folders = []
        self.categories = []
        
        # 遍历malignant和benign文件夹
        for category in ['malignant', 'benign']:
            category_path = os.path.join(data_path, category)
            if not os.path.exists(category_path):
                continue
                
            # 获取所有视频文件夹
            video_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
            
            for video_folder in video_folders:
                video_path = os.path.join(category_path, video_folder)
                # 获取视频文件夹中的所有图像文件
                image_files = glob.glob(os.path.join(video_path, '*.png')) + \
                             glob.glob(os.path.join(video_path, '*.jpg')) + \
                             glob.glob(os.path.join(video_path, '*.jpeg'))
                
                for image_file in image_files:
                    self.image_paths.append(image_file)
                    self.video_folders.append(video_folder)
                    self.categories.append(category)
        
        print(f"加载了 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        point_label = 1

        # 获取图像路径和相关信息
        img_path = self.image_paths[index]
        video_folder = self.video_folders[index]
        category = self.categories[index]
        
        # 获取图像文件名（不带扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        # 创建一个空白的mask作为占位符，因为数据集中没有真实的标签
        # 后续可以使用UNet生成标签
        dummy_gt = Image.new('L', (self.img_size, self.img_size), 0)
        
        # 调整图像大小
        newsize = (self.img_size, self.img_size)
        img = img.resize(newsize)
        
        # 根据提示类型生成点或框
        if self.prompt == 'click' or self.prompt == 'random_click':
            # 生成一个随机点作为占位符，后续可以使用真实的点
            if self.prompt == 'click':
                # 使用中心点
                h, w = self.img_size, self.img_size
                pt = np.array([h//2, w//2])
            else:
                # 使用随机点
                pt = np.array([np.random.randint(0, self.img_size), np.random.randint(0, self.img_size)])
            
            box_cup = np.array([0,0,0,0])
        elif self.prompt == 'box' or self.prompt == 'central_box':
            pt = np.array([0,0])
            if self.prompt == 'central_box':
                # 使用中心框
                h, w = self.img_size, self.img_size
                box_size = min(h, w) // 4
                box_cup = np.array([h//2 - box_size, w//2 - box_size, h//2 + box_size, w//2 + box_size])
            else:
                # 使用随机框
                h, w = self.img_size, self.img_size
                x1 = np.random.randint(0, h//2)
                y1 = np.random.randint(0, w//2)
                x2 = np.random.randint(h//2, h)
                y2 = np.random.randint(w//2, w)
                box_cup = np.array([x1, y1, x2, y2])
        else:
            pt = np.array([0,0])
            box_cup = np.array([0,0,0,0])
        
        # 应用变换
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                dummy_gt = self.transform_msk(dummy_gt).int()
        
        # 构建元数据
        image_meta_dict = {
            'filename_or_obj': img_name,
            'video_folder': video_folder,
            'category': category
        }
        
        return {
            'image': img,
            'label': dummy_gt,  # 使用空白标签作为占位符
            'pt_image': dummy_gt.clone() if isinstance(dummy_gt, torch.Tensor) else dummy_gt,  # 占位符
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }