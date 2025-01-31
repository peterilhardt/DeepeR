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
parser.add_argument('--train-val-split', default=85, type=int,
                    help='percentage of data to use for training (default: 85)')
parser.add_argument('--id', default='final', type=str,
                    help="Unique ID for the model or run")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--network', default='Hyperspectral_RCAN', type=str,
                    help='network')
parser.add_argument('--lam', default=100, type=int,
                    help='lambda (default: 100)')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-5)', dest='lr')
parser.add_argument('--base-lr', '--base-learning-rate', default=1e-7, type=float,
                    help='base learning rate (default: 1e-7)')
parser.add_argument('--scheduler', default='constant-lr', type=str,
                    help='scheduler')
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
                    help='number of nodes for distributed training '
                         '(-1 to get from environment)')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training ' 
                         '(-1 to get from environment)') 
parser.add_argument('--dist-url', default='tcp://localhost:12355', type=str,
                    help='url used to set up distributed training (IP address and port of '
                         'principal node - can use env:// to get from environment)')
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

    if args.dist_url == "env://" and args.world_size == -1: # env:// refers to environment
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

    if args.use_gpu:
        print("Use GPU: {} for training".format(args.gpu))
    else:
        print("Use CPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            #args.rank = int(os.environ['SLURM_NODEID'])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        print('Process:', args.rank)

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # dist.init_process_group(backend=args.dist_backend,
        #                         world_size=args.world_size, rank=args.rank)

        # want each node to have the same starting weights, which are random
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
        else:
            random.seed(0)
            torch.manual_seed(0)
    
    # ----------------------------------------------------------------------------------------
    # Create model(s) and send to device(s)
    # ----------------------------------------------------------------------------------------
    scale = args.hr_image_size // args.lr_image_size
    net = model.Hyperspectral_RCAN(args.spectrum_len, scale).float()
    args.device = torch.device('cuda:{}'.format(args.gpu)) if args.use_gpu else torch.device('cpu')
    
    if args.distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        #args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        else:
            net.to(args.device)
            net = torch.nn.parallel.DistributedDataParallel(net)
            #net = torch.nn.DataParallel(net).to(args.device) 
    else:
        args.rank = 0  # to make sure the model gets saved later
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
        net.to(args.device)
        #net = torch.nn.parallel.DistributedDataParallel(net)
    # else:
    #     net = nn.DataParallel(net).to(args.device)
       
    # ----------------------------------------------------------------------------------------
    # Define dataset path and data splits
    # ----------------------------------------------------------------------------------------    
    #dataset_path = "\Path\To\Dataset\"
    assert os.path.exists(args.dataset), 'Could not find path to training data file!'
    image_ids_csv = pd.read_csv(args.dataset)

    image_ids = image_ids_csv["id"].values

    train_split = round(args.train_val_split / 100 * len(image_ids))
    val_split = round((100 - args.train_val_split) / 100 * len(image_ids))
    #test_split = round(0.05 * len(image_ids))
    train_ids = image_ids[:train_split]
    val_ids = image_ids[train_split:train_split+val_split]
    #test_ids = image_ids[train_split+val_split:]

    # ----------------------------------------------------------------------------------------
    # Create datasets and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Train = dataset.RamanImageDataset(train_ids, os.path.dirname(args.dataset), batch_size = args.batch_size, 
                                                    hr_image_size = args.hr_image_size, lr_image_size = args.lr_image_size,
                                                    spectrum_len = args.spectrum_len, spectrum_shift = 0.1, spectrum_flip = True, 
                                                    horizontal_flip = True, vertical_flip = True, rotate = True, patch = True, mixup = True)

    Raman_Dataset_Val = dataset.RamanImageDataset(val_ids, os.path.dirname(args.dataset), batch_size = args.batch_size, 
                                                    hr_image_size = args.hr_image_size, lr_image_size = args.lr_image_size,
                                                    spectrum_len = args.spectrum_len)

    if args.distributed:
        # ensures that each process gets different data from the batch
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            Raman_Dataset_Train, num_replicas = args.world_size, rank = args.rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            Raman_Dataset_Val, num_replicas = args.world_size, rank = args.rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(Raman_Dataset_Train, batch_size = args.batch_size, 
        shuffle = not args.distributed, num_workers = args.workers, sampler = train_sampler)
    val_loader = DataLoader(Raman_Dataset_Val, batch_size = args.batch_size, shuffle = False, 
        num_workers = args.workers, sampler = val_sampler)

    # ----------------------------------------------------------------------------------------
    # Define criterion(s), optimizer(s), and scheduler(s)
    # ----------------------------------------------------------------------------------------

    # ------------Criterion------------
    criterion = nn.L1Loss().to(args.device)
    criterion_MSE = nn.MSELoss().to(args.device)

    # ------------Optimizer------------
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr = args.lr)
    elif args.optimizer == "adamW":
        optimizer = optim.AdamW(net.parameters(), lr = args.lr)
    else: # Adam
        optimizer = optim.Adam(net.parameters(), lr = args.lr)

    # ------------Scheduler------------
    if args.scheduler == "decay-lr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    elif args.scheduler == "multiplicative-lr":
        lmbda = lambda epoch: 0.985
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif args.scheduler == "cyclic-lr":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = args.base_lr, max_lr = args.lr, mode = 'triangular2', cycle_momentum = False)
    elif args.scheduler == "one-cycle-lr":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, cycle_momentum = False)
    else: # constant-lr
        scheduler = None
    
    if args.rank == 0:
        print('Started Training')
        print('Training Details:')
        print('Network:         {}'.format(args.network))
        print('Epochs:          {}'.format(args.epochs))
        print('Batch Size:      {}'.format(args.batch_size))
        print('Optimizer:       {}'.format(args.optimizer))
        print('Scheduler:       {}'.format(args.scheduler))
        print('Learning Rate:   {}'.format(args.lr))
        print('Spectrum Length: {}'.format(args.spectrum_len))

        # the steps below tend to fail with distributed processes, so restrict them to one process
        date = datetime.datetime.now().strftime("%Y_%m_%d")

        if not os.path.exists('./models'):
            os.mkdir('models')
        if not os.path.exists('./runs'):
            os.mkdir('runs')
        log_dir = "runs/{}_{}_{}_{}_{}x_{}".format(date, args.optimizer, args.scheduler, args.network, scale, args.id)
        models_dir = "models/{}_{}_{}_{}_{}x_{}.pt".format(date, args.optimizer, args.scheduler, args.network, scale, args.id)

        writer = SummaryWriter(log_dir = log_dir)

    for epoch in range(args.epochs):
        train_loss, train_psnr, train_ssim = train(train_loader, net, optimizer, scheduler, criterion, criterion_MSE, epoch, args)
        valid_loss, valid_psnr, valid_ssim = validate(val_loader, net, criterion_MSE, args)
        if args.scheduler != "cyclic-lr" and args.scheduler != "one-cycle-lr" and args.scheduler != "constant-lr":
            scheduler.step()

        if args.rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', valid_loss, epoch)
            writer.add_scalar('PSNR/train', train_psnr, epoch)
            writer.add_scalar('PSNR/val', valid_psnr, epoch)
            writer.add_scalar('SSIM/train', train_ssim, epoch)
            writer.add_scalar('SSIM/val', valid_ssim, epoch)
                        
        # only want to save one copy of the model (weights are the same on each process)
        # don't want each process trying to save weights over each other
        if args.rank == 0:  
            if args.multiprocessing_distributed:  # multiprocessing adds "module" to each layer name
                torch.save(net.module.state_dict(), models_dir)
            else:
                torch.save(net.state_dict(), models_dir)
    print('Finished Training')

def train(dataloader, net, optimizer, scheduler, criterion, criterion_MSE, epoch, args):
    
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    psnr = utilities.AverageMeter('PSNR', ':.4f')
    ssim = utilities.AverageMeter('SSIM', ':.4f')
    progress = utilities.ProgressMeter(len(dataloader), [batch_time, psnr, ssim], prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, data in enumerate(dataloader):
        inputs = data['input_image']
        inputs = inputs.float()
        inputs = inputs.to(args.device)
        target = data['output_image']
        target = target.float()
        target = target.to(args.device)

        output = net(inputs)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if args.scheduler == "cyclic-lr" or args.scheduler == "one-cycle-lr":
            scheduler.step()        

        loss_MSE = criterion_MSE(output, target)
        losses.update(loss_MSE.item(), inputs.size(0)) 
        
        psnr_batch = utilities.calc_psnr(output, target)
        psnr.update(psnr_batch, inputs.size(0))
        
        ssim_batch = utilities.calc_ssim(output, target)
        ssim.update(ssim_batch, inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and args.rank == 0:
            progress.display(i)

    return losses.avg, psnr.avg, ssim.avg


def validate(dataloader, net, criterion_MSE, args):
    
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    psnr = utilities.AverageMeter('PSNR', ':.4f')
    ssim = utilities.AverageMeter('SSIM', ':.4f')
    progress = utilities.ProgressMeter(len(dataloader), [batch_time, psnr, ssim], prefix='Validation: ')

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(dataloader):
            inputs = data['input_image']
            inputs = inputs.float()
            inputs = inputs.to(args.device)
            target = data['output_image']
            target = target.float()
            target = target.to(args.device)

            output = net(inputs)

            loss_MSE = criterion_MSE(output, target)
            losses.update(loss_MSE.item(), inputs.size(0)) 

            psnr_batch = utilities.calc_psnr(output, target)
            psnr.update(psnr_batch, inputs.size(0))

            ssim_batch = utilities.calc_ssim(output, target)
            ssim.update(ssim_batch, inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0 and args.rank == 0:
                progress.display(i)

    return losses.avg, psnr.avg, ssim.avg


if __name__ == '__main__':
    main()