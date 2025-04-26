import argparse
import os
import csv
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *
import cv2

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.ones([1]).to(device) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    threshold = (0.3, 0.5, 0.7)
    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels.shape)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                    
            imge= net.image_encoder(imgs)
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )
                
            # Resize to the ordered output size
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, threshold, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    use_gpu = not args.no_gpu
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    tot_global = 0 # global sum metric, for return global loss and mean metric
    mix_res_sq = (0,)*args.multimask_output*2    # 累加 IoU² 和 Dice²
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    # threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    threshold = (0.3, 0.5, 0.7)
    if use_gpu:
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    else:
        GPUdevice = torch.device('cpu') 
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    # define csv file's path
    csv_folder = args.path_helper['log_path']
    individual_metrics_file = os.path.join(csv_folder, "individual_metrics.csv")
    # write title row
    with open(individual_metrics_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Dice", "IoU"])
    # Clarify: pt means prompt point, pt_images means prompt image, box means prompt box
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            pt_imagesw = pack.get('pt_image', None)
            if pt_imagesw is not None:
                pt_imagesw = pt_imagesw.to(dtype=torch.float32, device=GPUdevice)

            boxw = None
            point_labels = pack.get('p_label', None)
            
            if args.prompt_type == 'box' or args.prompt_type == 'central_box':
                boxw = pack['box']
            
            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    box = boxw
                    pt = ptw
                if pt_imagesw is not None:
                    pt_images = pt_imagesw[...,buoy:buoy + evl_ch]
                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    if pt_imagesw is not None:
                        pt_images = rearrange(pt_images, 'b c h w d -> (b d) c h w ')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    if pt_imagesw is not None:
                        pt_images = torchvision.transforms.Resize((args.image_size,args.image_size))(pt_images)
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt
                showbox = box

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels.shape)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch)

                if args.prompt_type == 'box' or args.prompt_type == 'central_box':
                    # 将 box_cup 转换为 PyTorch 张量，并转移到指定的 GPU 设备
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=GPUdevice)
                    # if(len(point_labels.shape)==1): # only one box prompt:
                    showbox = showbox[None, None, :]
                    # 如果需要将其转换为 [1, 1, 4] 形状（假设只有一个 box 坐标）
                    box_torch = box_torch[None, None, :]  # 维度变为 [1, 1, 4]

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    if args.prompt_type == 'click' or args.prompt_type == 'random_click':
                        if args.net == 'sam' or args.net == 'mobile_sam':
                            se, de = net.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )
                        elif args.net == "efficient_sam":
                            coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                            se = net.prompt_encoder(
                                coords=coords_torch,
                                labels=labels_torch,
                            )

                        if args.net == 'sam':
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                        elif args.net == 'mobile_sam':
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=False,
                            )
                        elif args.net == "efficient_sam":
                            se = se.view(
                                se.shape[0],
                                1,
                                se.shape[1],
                                se.shape[2],
                            )
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                multimask_output=False,
                            )

                        # Resize to the ordered output size
                        pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                        tot += lossfunc(pred, masks)

                    if args.prompt_type == 'box' or args.prompt_type == 'central_box':
                        if args.net == 'sam' or args.net == 'mobile_sam':
                            se, de = net.prompt_encoder(
                                points=None,
                                boxes=box_torch,
                                masks=None,
                            )
                        elif args.net == "efficient_sam":
                            coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                            se = net.prompt_encoder(
                                coords=coords_torch,
                                labels=labels_torch,
                            )

                        if args.net == 'sam':
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                        elif args.net == 'mobile_sam':
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=False,
                            )
                        elif args.net == "efficient_sam":
                            se = se.view(
                                se.shape[0],
                                1,
                                se.shape[1],
                                se.shape[2],
                            )
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                multimask_output=False,
                            )

                        # Resize to the ordered output size
                        pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                        tot += lossfunc(pred, masks)


                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        if pt_imagesw is not None:
                            if args.prompt_type == 'click' or args.prompt_type == 'random_click':
                                vis_image(imgs,pred,masks, threshold, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp, pt_images=pt_images)
                            elif args.prompt_type == 'box' or args.prompt_type == 'central_box':
                                vis_image(imgs,pred,masks, threshold, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, boxes = showbox, pt_images=pt_images)
                        else:
                            vis_image(imgs,pred, masks, threshold, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

                        # 保存预测掩码为单独的文件
                        for b_idx in range(pred.shape[0]):  # 遍历批次中的每个样本
                            # 从元数据中获取图像名称
                            if isinstance(name, list) and len(name) > b_idx:
                                img_name = name[b_idx].split('/')[-1].split('.')[0]
                            elif 'filename_or_obj' in pack['image_meta_dict']:
                                # 从元数据获取文件名
                                filename = pack['image_meta_dict']['filename_or_obj']
                                if isinstance(filename, list) and len(filename) > b_idx:
                                    img_name = filename[b_idx]
                                else:
                                    img_name = filename
                            
                            # 创建唯一的文件名
                            mask_name = f"{img_name}.png"
                            
                            # 检查是否为MICCAI数据集，如果是，则按视频文件夹保存
                            if args.dataset == 'miccai':
                                # 从图像元数据中获取视频文件夹信息
                                video_folder = pack['image_meta_dict']['video_folder']
                                category = pack['image_meta_dict']['category']
                                
                                # 处理列表情况
                                if isinstance(video_folder, list) and b_idx < len(video_folder):
                                    video_folder = video_folder[b_idx]
                                if isinstance(category, list) and b_idx < len(category):
                                    category = category[b_idx]
                                
                                # 创建保存路径：masks/category/video_folder/
                                mask_dir = os.path.join(args.path_helper['sample_path'], 'masks', category, video_folder)
                                os.makedirs(mask_dir, exist_ok=True)
                                
                                mask_path = os.path.join(mask_dir, mask_name)
                            else:
                                # 原来的保存逻辑
                                mask_path = os.path.join(args.path_helper['sample_path'], 'masks', mask_name)
                                
                                # 确保目录存在
                                os.makedirs(os.path.join(args.path_helper['sample_path'], 'masks'), exist_ok=True)
                            
                            # 将预测掩码转换为numpy数组并保存
                            # Method 0: binary
                            pred_mask = (pred[b_idx, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

                            # Method 1: original
                            # pred_mask = (pred[b_idx, 0].cpu().numpy()).astype(np.uint8) * 255

                            # Method 2: Sigmoid (not working)
                            # if torch.max(pred[b_idx, 0]) > 1 or torch.min(pred[b_idx, 0]) < 0:
                            #     pred[b_idx, 0] = torch.sigmoid(pred[b_idx, 0])
                            # pred_mask = (pred[b_idx, 0].cpu().numpy()).astype(np.uint8) * 255

                            cv2.imwrite(mask_path, pred_mask)
                    
                    for na in name[:2]:
                        img_name = na.split('/')[-1].split('.')[0]
                    temp = eval_seg(pred, masks, threshold, img_name, individual_metrics_file)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                     # 累加 IoU² 和 Dice²
                    mix_res_sq = tuple(sum(a) for a in zip(mix_res_sq, (temp[0]**2, temp[1]**2)))

            pbar.update()
    # 计算均值
    iou_mean, dice_mean = tuple(a / n_val for a in mix_res)
    # 计算方差: Var(X) = E(X²) - E(X)²
    iou_var = (mix_res_sq[0] / n_val) - (iou_mean ** 2)
    dice_var = (mix_res_sq[1] / n_val) - (dice_mean ** 2)

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot / n_val, (iou_mean, dice_mean, iou_var, dice_var)

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )