# train.py
#!/usr/bin/env	python3

""" One single sample evaluation
    Yufeng Deng
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function


def main():
    args = cfg.parse_args()
    if args.dataset == 'refuge' or args.dataset == 'refuge2':
        args.data_path = '../dataset'

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)


    if args.weights != 0:
        '''load adapted model'''
        assert args.weights != 0
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        state_dict = checkpoint['state_dict']
    else :
        '''load pretrained model'''
        checkpoint_file = os.path.join(args.sam_ckpt)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = 0
        state_dict = checkpoint

    if args.distributed != 'none':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            name = 'module.' + k
            new_state_dict[name] = v
        # load params
    else:
        new_state_dict = state_dict
    
    net.load_state_dict(new_state_dict)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')


    '''segmentation data'''
    nice_train_loader, nice_test_loader = get_dataloader(args)

    # set the model to evaluation mode
    net.eval()

    # start evaluation
    time_start = time.time()
    logger.info("Starting evaluation...")

    if args.dataset != 'REFUGE':
        tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net, writer)
        logger.info(f'Evaluation completed. Total score: {tol}, IOU: {eiou}, DICE: {edice}.')
    else:
        tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, start_epoch, net, writer)
        logger.info(
            f'Evaluation completed. Total score: {tol}, '
            f'IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, '
            f'DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc}.'
        )

    time_end = time.time()
    logger.info(f"Evaluation time: {time_end - time_start:.2f} seconds.")
    writer.close()


if __name__ == '__main__':
    main()
