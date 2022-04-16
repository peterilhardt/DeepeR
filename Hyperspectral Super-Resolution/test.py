import os
import sys
import random
import datetime
import time
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import math
from skimage.metrics import structural_similarity as sk_ssim

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import model, dataset, utilities

parser = argparse.ArgumentParser(description='HyRISR Training')

parser.add_argument('--dataset', default='', type=str, 
                    help='path to CSV listing image IDs to use as dataset '
                         '(images should be in same location)')
parser.add_argument('--model', default='', type=str, 
                    help='path to model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--network', default='Hyperspectral_RCAN', type=str,
                    help='network')
parser.add_argument('--lr-image-size', default=16, type=int,
                    help='low resolution image size (default: 16)')
parser.add_argument('--hr-image-size', default=64, type=int,
                    help='high resolution image size (default: 64)')
parser.add_argument('--spectrum-len', default=500, type=int,
                    help='spectrum length (default: 500)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use (use -1 for CPU)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12355', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend (use nccl for GPUs on Linux, gloo otherwise)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.use_gpu = torch.cuda.is_available() and args.gpu != -1  # -1 indicates want to use CPU
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    cores = torch.cuda.device_count() if args.use_gpu else torch.get_num_threads()  # mp.cpu_count()

    if not args.use_gpu:  
        args.dist_backend = 'gloo'  # nccl doesn't work with CPUs
        args.gpu = 0

    if args.multiprocessing_distributed:
        args.world_size = cores * args.world_size
        mp.spawn(main_worker, nprocs=cores, args=(cores, args))
    else:
        main_worker(args.gpu, cores, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        if args.use_gpu:
            print("Use GPU: {} for training".format(args.gpu))
        else:
            print("Use CPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # dist.init_process_group(backend=args.dist_backend,
        #                         world_size=args.world_size, rank=args.rank)
    
    # ----------------------------------------------------------------------------------------
    # Create model(s) and send to device(s)
    # ----------------------------------------------------------------------------------------
    scale = args.hr_image_size // args.lr_image_size
    net = model.Hyperspectral_RCAN(args.spectrum_len, scale).float()
    args.device = torch.device('cuda:{}'.format(args.gpu)) if args.use_gpu else torch.device('cpu')

    # if scale == 2:
    #     net.load_state_dict(torch.load('RCAN_2x.pt'))
    # elif scale == 3:
    #     net.load_state_dict(torch.load('RCAN_3x.pt'))
    # else: #scale == 4
    #     net.load_state_dict(torch.load('RCAN_4x.pt'))

    print('Model to be used:', args.model)
    assert os.path.exists(args.model), 'Could not find model path!'
    net.load_state_dict(torch.load(args.model))

    if args.distributed:
        if args.gpu is not None:
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            
            if args.use_gpu:
                torch.cuda.set_device(args.gpu)
                net.cuda(args.gpu)
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
            else:
                net.to(args.device)
                net = torch.nn.parallel.DistributedDataParallel(net)
        else:
            net.to(args.device)
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
        net.to(args.device)
    else:
        net = nn.DataParallel(net).to(args.device)
       
    # ----------------------------------------------------------------------------------------
    # Define dataset path and data splits
    # ----------------------------------------------------------------------------------------    
    #dataset_path = "\Path\To\Dataset\"
    assert os.path.exists(args.dataset), 'Could not find path to training data file!'
    image_ids_csv = pd.read_csv(args.dataset)

    image_ids = image_ids_csv["id"].values

    # ----------------------------------------------------------------------------------------
    # Create datasets and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Test = dataset.RamanImageDataset(image_ids, os.path.dirname(args.dataset), batch_size = args.batch_size, 
                                                    hr_image_size = args.hr_image_size, lr_image_size = args.lr_image_size,
                                                    spectrum_len = args.spectrum_len)

    test_loader = DataLoader(Raman_Dataset_Test, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)

    # ----------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------
    RCAN_PSNR, Bicubic_PSNR, Nearest_PSNR, RCAN_SSIM, Bicubic_SSIM, Nearest_SSIM, RCAN_MSE, Bicubic_MSE, Nearest_MSE = evaluate(test_loader, net, scale, args)

def evaluate(dataloader, net, scale, args):
    
    psnr = utilities.AverageMeter('PSNR', ':.4f')
    ssim = utilities.AverageMeter('SSIM', ':.4f')
    mse_NN = utilities.AverageMeter('MSE', ':.4f')
    psnr_bicubic = utilities.AverageMeter('PSNR_Bicubic', ':.4f')
    ssim_bicubic = utilities.AverageMeter('SSIM_Bicubic', ':.4f')
    mse_bicubic = utilities.AverageMeter('MSE_Bicubic', ':.4f')
    psnr_nearest_neighbours = utilities.AverageMeter('PSNR_Nearest_Neighbours', ':.4f')
    ssim_nearest_neighbours = utilities.AverageMeter('SSIM_Nearest_Neighbours', ':.4f')
    mse_nearest_neighbours = utilities.AverageMeter('MSE_Nearest_Neighbours', ':.4f')
    
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # measure data loading time
            x = data['input_image']
            inputs = x.float()
            inputs = inputs.to(args.device)
            y = data['output_image']
            target = y.float()
            target = target.to(args.device)

            # compute output
            output = net(inputs)

            x2 = np.squeeze(x.numpy())
            y2 = np.squeeze(y.numpy())

            seq = (1,1,scale,scale) if x2.ndim > 3 else (1,scale,scale)

            nearest_neighbours = scipy.ndimage.zoom(x2, seq, order=0)
            bicubic = scipy.ndimage.zoom(x2, seq, order=3)
                            
            bicubic = torch.from_numpy(bicubic)
            bicubic = bicubic.to(args.device)
            
            nearest_neighbours = torch.from_numpy(nearest_neighbours)
            nearest_neighbours = nearest_neighbours.to(args.device)

            # Nearest neighbours
            psnr_batch_nearest_neighbours = utilities.calc_psnr(nearest_neighbours, target.squeeze())
            psnr_nearest_neighbours.update(psnr_batch_nearest_neighbours, inputs.size(0))

            ssim_batch_nearest_neighbours = utilities.calc_ssim(nearest_neighbours, target.squeeze())
            ssim_nearest_neighbours.update(ssim_batch_nearest_neighbours, inputs.size(0))

            mse_batch_nearest_neighbours = nn.MSELoss()(nearest_neighbours, target.squeeze())
            mse_nearest_neighbours.update(mse_batch_nearest_neighbours, inputs.size(0))
            
            # Bicubic
            psnr_batch_bicubic = utilities.calc_psnr(bicubic, target.squeeze())
            psnr_bicubic.update(psnr_batch_bicubic, inputs.size(0))

            ssim_batch_bicubic = utilities.calc_ssim(bicubic, target.squeeze())
            ssim_bicubic.update(ssim_batch_bicubic, inputs.size(0))

            mse_batch_bicubic = nn.MSELoss()(bicubic, target.squeeze())
            mse_bicubic.update(mse_batch_bicubic, inputs.size(0))
            
            # Neural network
            psnr_batch = utilities.calc_psnr(output, target)
            psnr.update(psnr_batch, inputs.size(0))

            ssim_batch = utilities.calc_ssim(output, target)
            ssim.update(ssim_batch, inputs.size(0))
            
            mse_batch = nn.MSELoss()(output, target)
            mse_NN.update(mse_batch, inputs.size(0))
            
    print("RCAN PSNR: {}    Bicubic PSNR: {}    Nearest Neighbours PSNR: {}".format(psnr.avg, psnr_bicubic.avg, psnr_nearest_neighbours.avg))
    print("RCAN SSIM: {}    Bicubic SSIM: {}    Nearest Neighbours SSIM: {}".format(ssim.avg, ssim_bicubic.avg, ssim_nearest_neighbours.avg))
    print("RCAN MSE:  {}    Bicubic MSE:  {}    Nearest Neighbours MSE:  {}".format(mse_NN.avg, mse_bicubic.avg, mse_nearest_neighbours.avg))
    return psnr.avg, psnr_bicubic.avg, psnr_nearest_neighbours.avg, ssim.avg, ssim_bicubic.avg, ssim_nearest_neighbours.avg, mse_NN.avg, mse_bicubic.avg, mse_nearest_neighbours.avg

if __name__ == '__main__':
    main()