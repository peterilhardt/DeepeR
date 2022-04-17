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

import model, dataset, utilities, PCA_denoise, wavelet_denoise


parser = argparse.ArgumentParser(description='DeNoiser Training')

parser.add_argument('--features', default='', type=str, 
                    help='path to training data features/inputs')
parser.add_argument('--labels', default='', type=str, 
                    help='path to training data labels')
parser.add_argument('--model', default='', type=str, 
                    help='path to model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch-norm', action='store_true',
                    help='apply batch norm')
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
        args.gpu = 0
        main_worker(args.gpu, cores, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.use_gpu:
        print("Use GPU: {} for training".format(args.gpu))
    else:
        print("Use CPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            #args.rank = int(os.environ["RANK"])
            args.rank = int(os.environ['SLURM_NODEID'])
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
    assert os.path.exists(args.model), 'Could not find model path!'
    net = model.ResUNet(3, args.batch_norm).float()
    net.load_state_dict(torch.load(args.model))
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
    else:
        args.rank = 0 
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
        net.to(args.device)
    # else:
    #     net.to(args.device)
    #     net = torch.nn.parallel.DistributedDataParallel(net)

    # ----------------------------------------------------------------------------------------
    # Define dataset path and data splits
    # ----------------------------------------------------------------------------------------    
    assert os.path.exists(args.features), 'Could not find path to training data features!'
    assert os.path.exists(args.labels), 'Could not find path to training data labels!'
    Input_Data = scipy.io.loadmat(args.features)
    Output_Data = scipy.io.loadmat(args.labels)

    Input = Input_Data[os.path.basename(os.path.splitext(args.features)[0])]
    Output = Output_Data[os.path.basename(os.path.splitext(args.labels)[0])]

    # ----------------------------------------------------------------------------------------
    # Create datasets (with augmentation) and dataloaders
    # ----------------------------------------------------------------------------------------
    Raman_Dataset_Test = dataset.RamanDataset(Input, Output, batch_size = args.batch_size, spectrum_len = args.spectrum_len)

    if args.distributed:
        # ensures that each process gets different data from the batch
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            Raman_Dataset_Test, num_replicas = args.world_size, rank = args.rank)
    else:
        test_sampler = None

    test_loader = DataLoader(Raman_Dataset_Test, batch_size = args.batch_size, shuffle = False, 
        num_workers = 0, pin_memory = True, sampler = test_sampler)

    # ----------------------------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------------------------
    MSE_NN, MSE_SG, MSE_PCA, MSE_wavelet = evaluate(test_loader, net, args)

def evaluate(dataloader, net, args):
    losses = utilities.AverageMeter('Loss', ':.4e')
    SG_loss = utilities.AverageMeter('Savitzky-Golay Loss', ':.4e')

    net.eval()
    
    all_x = []
    all_y = []
    MSE_SG = []
    MSE_PCA = []
    MSE_wavelet = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x = data['input_spectrum']
            inputs = x.float()
            inputs = inputs.to(args.device)
            y = data['output_spectrum']
            target = y.float()
            target = target.to(args.device)
            
            x = np.squeeze(x.numpy())
            y = np.squeeze(y.numpy())

            all_x.append(x)
            all_y.append(y)

            output = net(inputs)
            loss = nn.MSELoss()(output, target)
            
            x_out = output.cpu().detach().numpy()
            x_out = np.squeeze(x_out)

            SGF_1_9 = scipy.signal.savgol_filter(x,9,1)
            MSE_SGF_1_9 = np.mean(np.mean(np.square(np.absolute(y - (SGF_1_9 - np.reshape(np.amin(SGF_1_9, axis = 1), (len(SGF_1_9),1)))))))
            MSE_SG.append(MSE_SGF_1_9)
            
            losses.update(loss.item(), inputs.size(0))

        if args.rank == 0:
            all_x = np.concatenate(all_x, axis = 0)
            all_y = np.concatenate(all_y, axis = 0)

            # get optimal PC and wavelet params for denoising along with best MSE
            MSE_PCA, num_PCs = PCA_denoise.get_optimal_pca(all_x, all_y, max_components = 10)
            print('Optimal number of PCs: {}'.format(num_PCs))
            MSE_wavelet, wave_type, wave_level = wavelet_denoise.get_optimal_wavelet(all_x, all_y, max_level = 2)
            print('Optimal wavelet type and level: {}, {}'.format(wave_type, wave_level))

            MSE_SG = np.mean(np.asarray(MSE_SG))
            print("Neural Network MSE: {}".format(losses.avg))
            print("Savitzky-Golay MSE: {}".format(MSE_SG))
            print("PCA MSE: {}".format(MSE_PCA))
            print("Wavelet MSE: {}".format(MSE_wavelet))
            print("Neural Network performed {0:.2f}x better than Savitzky-Golay".format(MSE_SG/losses.avg))

    return losses.avg, MSE_SG, MSE_PCA, MSE_wavelet


if __name__ == '__main__':
    main()
