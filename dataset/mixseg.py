import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click, rect_box, central_click, central_box

class MIXSEG(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, 
                 mode='test', prompt='click', test_dataset=None, prompt_source='unet'):
        """
        参数说明：
        - args: 包含 image_size 等属性的参数对象
        - data_path: BUS_Mix_Seg 文件夹路径
        - transform: 图像增强/变换方法
        - transform_msk: 标签及提示图像的变换方法
        - mode: "train" 或 "test"
        - prompt: 指定的提示方式，如 'click', 'box', 'random_click', 'central_box'
        - test_dataset: 测试数据集名称，例如 "BUSI"。也可以是逗号分隔的多个数据集，例如 "BUSI,BUET,STU"
                     在训练模式下，将排除这些数据集；在测试模式下，只使用这些数据集的数据
        - prompt_source: 提示来源，目前与 'unet' 等方式保持一致（本代码中未做特殊处理）
        """
        self.data_path = data_path
        self.mode = mode
        self.prompt = args.prompt_type
        
        # 处理测试数据集，支持多个数据集输入（逗号分隔）
        if args.test_dataset and ',' in args.test_dataset:
            self.test_dataset = [ds.strip() for ds in args.test_dataset.split(',')]
        else:
            self.test_dataset = [args.test_dataset] if args.test_dataset else None
            
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt_source = prompt_source

        # 构建 images、labels 和 prompts 的文件字典，键为文件名（去掉扩展名），值为完整路径
        self.image_dict = {os.path.splitext(f)[0]: os.path.join(data_path, 'images', f)
                           for f in os.listdir(os.path.join(data_path, 'images'))}
        self.label_dict = {os.path.splitext(f)[0]: os.path.join(data_path, 'labels', f)
                           for f in os.listdir(os.path.join(data_path, 'labels'))}
        if mode == 'train':
            self.prompt_dict = self.label_dict
        else:
            self.prompt_dict = self.label_dict
            # # 如果提示方式为 'random_click' 或 'central_box'，则直接使用图像作为提示，否则加载 prompts 文件夹中的图像
            # if self.prompt in ['random_click', 'central_box']:
            #     self.prompt_dict = self.image_dict
            # else:
            #     self.prompt_dict = {os.path.splitext(f)[0]: os.path.join(data_path, 'prompts', f)
            #                         for f in os.listdir(os.path.join(data_path, 'prompts'))}

        # 取 images 和 labels 的交集，确保每个样本都有对应的图像和标签
        keys = set(self.image_dict.keys()) & set(self.label_dict.keys())

        # 根据 mode 和 test_dataset 参数筛选文件：
        if self.test_dataset is not None:
            if self.mode == 'train':
                # 训练模式下，排除所有在 test_dataset 列表中的数据集
                keys = {k for k in keys if k.split('_')[0] not in self.test_dataset}
            elif self.mode == 'test':
                # 测试模式下，只保留在 test_dataset 列表中的数据集
                keys = {k for k in keys if k.split('_')[0] in self.test_dataset}
        
        self.name_list = list(keys)

        print(f"[DEBUG] mode={self.mode}, test_dataset={self.test_dataset}")
        print(f"[DEBUG] Total before filter: {len(self.image_dict)}")
        print(f"[DEBUG] Keys after filtering: {len(keys)}")
        print(f"[DEBUG] Example keys: {list(keys)[:5]}")


    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        point_label = 1
        # 根据索引获取文件名
        name = self.name_list[index]
        img_path = self.image_dict[name]
        gt_path = self.label_dict[name]
        prompt_path = self.prompt_dict.get(name)

        # 打开图像：图像转为 RGB，标签和提示图像转为灰度图（L模式）
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        prompt_image = Image.open(prompt_path).convert('L')

        # 调整标签和提示图像大小
        newsize = (self.img_size, self.img_size)
        gt = gt.resize(newsize)
        prompt_image = prompt_image.resize(newsize)

        # 根据提示方式生成提示信息（点坐标或边界框）
        if self.prompt == 'click':
            # 此处示例使用 central_click，可根据需要选择 random_click
            point_label, pt = central_click(np.array(prompt_image) / 255, point_label)
            box_cup = np.array([0, 0, 0, 0])
        elif self.prompt == 'box':
            pt = np.array([0, 0])
            prompt_image_np = np.array(prompt_image)
            box_cup = rect_box(prompt_image_np)
        elif self.prompt == 'random_click':
            point_label, pt = random_click(np.array(prompt_image) / 255, point_label)
            box_cup = np.array([0, 0, 0, 0])
        elif self.prompt == 'central_box':
            pt = np.array([0, 0])
            prompt_image_np = np.array(prompt_image)
            box_cup = central_box(prompt_image_np)
        else:
            pt = np.array([0, 0])
            box_cup = np.array([0, 0, 0, 0])

        # 如果定义了 transform，则对图像、标签和提示图像同时进行相同的变换
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            if self.transform_msk:
                gt = self.transform_msk(gt).int()
                prompt_image = self.transform_msk(prompt_image).int()
        
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'label': gt,
            'pt_image': prompt_image,
            'p_label': point_label,
            'pt': pt,
            'box': box_cup,
            'image_meta_dict': image_meta_dict,
        }
