import os
import glob
import subprocess
import numpy as np
import scipy.io
import datetime

DATA_BASE_PATH = r'../../data/spectral_denoising_data'
FOLDS = 11

ARGS = {
    'train_val_split': 90,
    'workers': 0,
    'epochs': 1,
    'start_epoch': 0,
    'batch_size': 256,
    'network': 'ResUNet',
    'optimizer': 'adam',
    'lr': 5e-4,
    'base_lr': 5e-6,
    'scheduler': 'one-cycle-lr',
    'batch_norm': True,
    'spectrum_len': 500,
    'seed': 42,
    'gpu': 0,
    'world_size': 1,
    'rank': 0,
    'dist_url': 'tcp://localhost:12355',
    'dist_backend': 'nccl',
    'multiprocessing_distributed': True
}


# def make_folds(folds = FOLDS, base_path = DATA_BASE_PATH):
#     train_inputs = scipy.io.loadmat(os.path.join(base_path, 'Train_Inputs.mat'))['Train_Inputs']
#     train_outputs = scipy.io.loadmat(os.path.join(base_path, 'Train_Outputs.mat'))['Train_Outputs']
#     test_inputs = scipy.io.loadmat(os.path.join(base_path, 'Test_Inputs.mat'))['Test_Inputs']
#     test_outputs = scipy.io.loadmat(os.path.join(base_path, 'Test_Outputs.mat'))['Test_Outputs']

#     all_inputs = np.concatenate([train_inputs, test_inputs], axis = 0)
#     all_outputs = np.concatenate([train_outputs, test_outputs], axis = 0)
#     n = len(all_inputs)
#     fold_size = n // folds

#     for i in range(folds):
#         inputs = all_inputs[(fold_size * i):(fold_size * (i+1)),:]
#         outputs = all_outputs[(fold_size * i):(fold_size * (i+1)),:]

#         name_i = 'fold{}_inputs'.format(i)
#         name_o = 'fold{}_outputs'.format(i)
#         scipy.io.savemat(os.path.join(base_path, name_i + '.mat'), {name_i: inputs})
#         scipy.io.savemat(os.path.join(base_path, name_o + '.mat'), {name_o: outputs})

def cleanup_folds(base_path = DATA_BASE_PATH, only_train = True):
    if only_train:
        files = glob.glob(os.path.join(base_path, 'fold*_train.mat'))
    else:
        files = glob.glob(os.path.join(base_path, 'fold*.mat'))
    for file in files:
        if os.path.exists(file):
            os.remove(file)

def make_cmd(feature_path, label_path, args, fold, train = True):
    new_args = args.copy()

    DATE = datetime.datetime.now().strftime("%Y_%m_%d")
    multiproc = new_args['multiprocessing_distributed']
    batch_norm = new_args['batch_norm']
    new_args['features'] = feature_path
    new_args['labels'] = label_path

    if not train:
        rm_list = ['train_val_split', 'epochs', 'start_epoch', 'lr', 'base_lr', 
                    'batch_norm', 'multiprocessing_distributed']
        for key in rm_list:
            new_args.pop(key)
        model = "models/{}_{}_{}_{}_{}.pt".format(DATE, new_args['optimizer'], new_args['scheduler'], 
                                                  new_args['network'], 'fold{}'.format(fold))
        new_args['model'] = model
        for key in ['optimizer', 'scheduler', 'network']:
            new_args.pop(key)
        cmd = 'python test.py '
    else:
        for key in ['batch_norm', 'multiprocessing_distributed']:
            new_args.pop(key)
        new_args['id'] = 'fold{}'.format(fold)
        cmd = 'python train.py '
    
    cmd += ' '.join(['--{} {}'.format(k.replace('_', '-'), str(v)) for k, v in new_args.items()])
    if batch_norm:
        cmd += ' --batch-norm'
    if multiproc:
        cmd += ' --multiprocessing-distributed'

    return cmd
        
def crossval_train(folds = FOLDS, base_path = DATA_BASE_PATH, args = ARGS):
    train_inputs = scipy.io.loadmat(os.path.join(base_path, 'Train_Inputs.mat'))['Train_Inputs']
    train_outputs = scipy.io.loadmat(os.path.join(base_path, 'Train_Outputs.mat'))['Train_Outputs']
    test_inputs = scipy.io.loadmat(os.path.join(base_path, 'Test_Inputs.mat'))['Test_Inputs']
    test_outputs = scipy.io.loadmat(os.path.join(base_path, 'Test_Outputs.mat'))['Test_Outputs']

    all_inputs = np.concatenate([train_inputs, test_inputs], axis = 0)
    all_outputs = np.concatenate([train_outputs, test_outputs], axis = 0)
    # all_inputs = test_inputs.copy()
    # all_outputs = test_outputs.copy()
    n = len(all_inputs)
    fold_size = n // folds

    for i in range(folds):
        # extract subset of inputs and outputs
        test_idx = np.arange((fold_size * i), (fold_size * (i+1)))
        test_inputs = all_inputs[test_idx,:]
        test_outputs = all_outputs[test_idx,:]
        train_inputs = np.delete(all_inputs, test_idx)
        train_outputs = np.delete(all_outputs, test_idx)

        # write new train and test sets to .mat files
        name_train_i = 'fold{}_inputs_train'.format(i)
        name_train_o = 'fold{}_outputs_train'.format(i)
        name_test_i = 'fold{}_inputs_test'.format(i)
        name_test_o = 'fold{}_outputs_test'.format(i)
        scipy.io.savemat(os.path.join(base_path, name_train_i + '.mat'), {name_train_i: train_inputs})
        scipy.io.savemat(os.path.join(base_path, name_train_o + '.mat'), {name_train_o: train_outputs})
        scipy.io.savemat(os.path.join(base_path, name_test_i + '.mat'), {name_test_i: test_inputs})
        scipy.io.savemat(os.path.join(base_path, name_test_o + '.mat'), {name_test_o: test_outputs})

        # train a model for each fold
        cmd = make_cmd(os.path.join(base_path, name_train_i + '.mat'), 
                        os.path.join(base_path, name_train_o + '.mat'),
                        args, i, train = True)
        
        print('Training model for fold {}'.format(i))
        subprocess.run(cmd.split(' '))

        # delete the training set to cleanup
        cleanup_folds(base_path, only_train = True)

def crossval_eval(folds = FOLDS, base_path = DATA_BASE_PATH, args = ARGS, delete_test_sets = False):
    for i in range(folds):
        # evaluate each model on its held-out test data
        name_test_i = 'fold{}_inputs_test'.format(i)
        name_test_o = 'fold{}_outputs_test'.format(i)

        cmd = make_cmd(os.path.join(base_path, name_test_i + '.mat'), 
                        os.path.join(base_path, name_test_o + '.mat'),
                        args, i, train = False)

        print('Evaluating model for fold {}'.format(i))
        subprocess.run(cmd.split(' '))

    # delete test data if needed
    if delete_test_sets:
        cleanup_folds(base_path, only_train = False)


if __name__ == '__main__':
    crossval_train(FOLDS, DATA_BASE_PATH, ARGS)
    #ARGS['multiprocessing_distributed'] = False
    crossval_eval(FOLDS, DATA_BASE_PATH, ARGS, delete_test_sets = True)
