import os
import random
import datetime
import time
import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import model, dataset, utilities

parser = argparse.ArgumentParser(description='Segmentation Training')

parser.add_argument('--dataset', default='', type=str, 
                    help='path to CSV listing image IDs to use as dataset '
                         '(images should be in same location)')
parser.add_argument('--labels', default='', type=str,
                    help='path to directory containing label files/masks '
                         '(should be PNG files with same names as dataset images)')
parser.add_argument('--model', default='', type=str, 
                    help='path to model')
parser.add_argument('--dims', default=2, type=int, 
                    help='image convolution dimension (2 or 3)')
parser.add_argument('--classes', default=2, type=int,
                    help='number of segmentation classes (not including background class)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--network', default='UNet', type=str,
                    help='network')
parser.add_argument('--image-size', default=64, type=int,
                    help='image size (default: 64)')
parser.add_argument('--spectrum-len', default=500, type=int,
                    help='spectrum length (default: 500)')
parser.add_argument('--normalization', default='', type=str,
                    help='normalization type '
                         '(None/BatchNorm/LayerNorm/InstanceNorm/GroupNorm; default None)')
parser.add_argument('--activation', default='ReLU', type=str,
                    help='activation type '
                         '(ReLU/LeakyReLU/PReLU; default ReLU)')
parser.add_argument('--residual', action='store_true',
                    help='whether to apply residual connections to U-Net conv. layers')
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

    #---------Activation---------------
    if args.activation == 'LeakyReLU':
        activation = nn.LeakyReLU()
    elif args.activation == 'PReLU':
        activation = nn.PReLU()
    else:
        activation = nn.ReLU()
    
    # ----------------------------------------------------------------------------------------
    # Create model(s) and send to device(s)
    # ----------------------------------------------------------------------------------------
    normalization = None if args.normalization in ['', 'None'] else args.normalization
    if args.dims == 2:
        net = model.UNet(args.dims, channels = args.spectrum_len, 
                        num_classes = args.classes, normalization = normalization, 
                        activation = activation, res = args.residual).float()
    elif args.dims == 3:
        net = model.UNet(args.dims, channels = 1, img_channels = args.spectrum_len, 
                        num_classes = args.classes, normalization = normalization, 
                        activation = activation, res = args.residual).float()
    else:
        raise ValueError('Unsupported image dimensionality for segmentation')
    args.device = torch.device('cuda:{}'.format(args.gpu)) if args.use_gpu else torch.device('cpu')

    print('Model to be used:', args.model)
    assert os.path.exists(args.model), 'Could not find model path!'
    net.load_state_dict(torch.load(args.model))

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
    else:
        args.rank = 0 
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
        net.to(args.device)
    # else:
    #     net = nn.DataParallel(net).to(args.device)
       
    # ----------------------------------------------------------------------------------------
    # Define dataset path and data splits
    # ----------------------------------------------------------------------------------------    
    assert os.path.exists(args.dataset), 'Could not find path to training data file!'
    assert not args.labels or os.path.exists(args.labels), 'Invalid path to label files provided!'

    image_ids_csv = pd.read_csv(args.dataset)
    image_ids = image_ids_csv["id"].values

    # ----------------------------------------------------------------------------------------
    # Create datasets and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Test = dataset.RamanImageDataset(image_ids, os.path.dirname(args.dataset), args.labels, 
                                                    batch_size = args.batch_size, image_size = args.image_size,
                                                    spectrum_len = args.spectrum_len)

    if args.distributed:
        # ensures that each process gets different data from the batch
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            Raman_Dataset_Test, num_replicas = args.world_size, rank = args.rank)
    else:
        test_sampler = None

    test_loader = DataLoader(Raman_Dataset_Test, batch_size = args.batch_size, shuffle = False, 
        num_workers = args.workers, sampler = test_sampler)

    # ----------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------
    test_loss = evaluate(test_loader, net, args)

def evaluate(dataloader, net, args):
    
    losses = utilities.AverageMeter('Loss', ':.4e')
    
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x = data['image']
            inputs = x.float()
            inputs = inputs.to(args.device)
            y = data['mask']
            target = y.long()
            target = target.to(args.device)

            output = net(inputs)
            
            loss = nn.CrossEntropyLoss()(output, target)
            losses.update(loss.item(), inputs.size(0))

    if args.rank == 0:        
        print("UNet Loss: {}".format(losses.avg))

    return losses.avg


if __name__ == '__main__':
    main()