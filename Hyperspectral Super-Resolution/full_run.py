import os
import pandas as pd
import glob
import subprocess
import numpy as np
import scipy.io
import datetime

DATA_BASE_PATH = r'../../data/hyperspectral_super_resolution_data'
TRAIN_PERCENT = 85
VAL_PERCENT = 10

ARGS = {
    'train_val_split': 89,
    'id': 'full',
    'workers': 0,
    'epochs': 1,
    'start_epoch': 0,
    'batch_size': 2,
    'network': 'Hyperspectral_RCAN',
    'lam': 100,
    'optimizer': 'adam',
    'lr': 1e-5,
    'base_lr': 1e-7,
    'scheduler': 'constant-lr',
    'lr_image_size': 32,
    'hr_image_size': 64,
    'spectrum_len': 500,
    'seed': 42,
    'gpu': 0,
    'world_size': 1,
    'rank': 0,
    'dist_url': 'tcp://localhost:12355',
    'dist_backend': 'nccl',
    'multiprocessing_distributed': True
}


def write_id_files(data_path = DATA_BASE_PATH, train_split = TRAIN_PERCENT, val_split = VAL_PERCENT):
    files = os.listdir(data_path)
    files = sorted([f[:-4] for f in files if f.split('.')[-1] == 'mat'])
    df = pd.DataFrame({'id': files})
    df = df.sample(frac = 1).reset_index(drop = True)  # shuffles the rows

    n = df.shape[0]
    split = int((train_split + val_split) / 100 * n)
    df_train_val = df.iloc[:split,:]
    df_test = df.iloc[split:,:]
    
    train_file = os.path.join(data_path, 'Train_Image_IDs.csv')
    test_file = os.path.join(data_path, 'Test_Image_IDs.csv')

    if not os.path.exists(train_file):
        df_train_val.to_csv(train_file, index = False)
    if not os.path.exists(test_file):
        df_test.to_csv(test_file, index = False)

def clear_id_files(data_path = DATA_BASE_PATH):
    train_file = os.path.join(data_path, 'Train_Image_IDs.csv')
    test_file = os.path.join(data_path, 'Test_Image_IDs.csv')
    for file in [train_file, test_file]:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            continue

def make_cmd(dataset_path, args, train = True):
    new_args = args.copy()

    scale = new_args['hr_image_size'] // new_args['lr_image_size']
    DATE = datetime.datetime.now().strftime("%Y_%m_%d")
    multiproc = new_args['multiprocessing_distributed']
    new_args['dataset'] = dataset_path

    if not train:
        rm_list = ['train_val_split', 'epochs', 'start_epoch', 'lr', 'base_lr', 
                    'lam', 'multiprocessing_distributed']
        for key in rm_list:
            new_args.pop(key)
        model = "models/{}_{}_{}_{}_{}x_{}.pt".format(DATE, new_args['optimizer'], new_args['scheduler'], 
                                                      new_args['network'], scale, new_args['id'])
        new_args['model'] = model
        for key in ['optimizer', 'scheduler', 'id']:
            new_args.pop(key)
        cmd = 'python test.py '
    else:
        new_args.pop('multiprocessing_distributed')
        cmd = 'python train.py '
    
    cmd += ' '.join(['--{} {}'.format(k.replace('_', '-'), str(v)) for k, v in new_args.items()])
    if multiproc:
        cmd += ' --multiprocessing-distributed'

    return cmd

def train(data_path = DATA_BASE_PATH, train_split = TRAIN_PERCENT, val_split = VAL_PERCENT, args = ARGS):
    write_id_files(data_path, train_split, val_split)
    dataset_path = os.path.join(DATA_BASE_PATH, 'Train_Image_IDs.csv')
    cmd = make_cmd(dataset_path, args, train = True)

    print('Training model')
    subprocess.run(cmd.split(' '))

def eval(data_path = DATA_BASE_PATH, args = ARGS, delete_id_files = True):
    dataset_path = os.path.join(DATA_BASE_PATH, 'Test_Image_IDs.csv')
    cmd = make_cmd(dataset_path, args, train = False)

    print('Evaluating model')
    subprocess.run(cmd.split(' '))

    if delete_id_files:
        clear_id_files(data_path)


if __name__ == '__main__':
    train(DATA_BASE_PATH, TRAIN_PERCENT, VAL_PERCENT, ARGS)
    eval(DATA_BASE_PATH, ARGS, delete_id_files = False)
    