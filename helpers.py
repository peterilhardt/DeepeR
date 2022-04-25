import os
import scipy.io
import scipy.ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as sk_ssim

def load_image(base_path, id_name):
    ''' Load a hyperspectral image '''
    input_path = os.path.join(base_path, id_name + ".mat")
    output_data = scipy.io.loadmat(input_path)
    output_values = list(output_data.values())
    output_image = output_values[3]
    return output_image

def load_spectra(base_path, file_name):
    ''' Load a spectra Matlab file '''
    data = scipy.io.loadmat(os.path.join(base_path, file_name))
    values = list(data.values())
    return values[3]

def plot_image(img, cmap = 'viridis', title = None, file = None, show = True):
    ''' Plot a single hyperspectral image '''
    if img.ndim == 3:
        img = img.mean(axis = 2)
    fig, ax = plt.subplots(1, 1, figsize = (7,5))
    ax.imshow(img, cmap = cmap, interpolation = 'nearest')
    ax.axis('off')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if file:
        plt.savefig(file)
    if show:
        plt.show()

def plot_images(img_list, cmap = 'viridis', labels = None, file = None, show = True):
    ''' Plot multiple hyperspectral images (in a list) side-by-side '''
    n = len(img_list)
    fig, ax = plt.subplots(1, n, figsize = (7 * n, 5), constrained_layout = True)
    for i in range(n):
        img = img_list[i]
        if img.ndim == 3:
            img = img.mean(axis = 2)
        ax[i].imshow(img, cmap = cmap, interpolation = 'nearest')
        ax[i].axis('off')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        if labels is not None:
            ax[i].set_title(labels[i])
    #fig.tight_layout()
    #fig.subplots_adjust(wspace = 0, hspace = 0)
    if file:
        plt.savefig(file = None)
    if show:
        plt.show()

def plot_spec(spec, wl = None, title = None, file = None, show = True):
    ''' Plot a single spectrum '''
    wave = wl if wl is not None else list(range(len(spec)))
    fig, ax = plt.subplots(1, 1, figsize = (8, 4))
    ax.plot(wave, spec)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if file:
        plt.savefig(file)
    if show:
        plt.show()

def plot_spec_overlay(spec, wl = None, labels = None, file = None, show = True):
    ''' Plot multiple spectra from an array over each other '''
    wave = wl if wl is not None else list(range(spec.shape[1]))
    if not labels:
        labels = ['Spectrum {}'.format(i+1) for i in range(len(spec))]
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    for i in range(len(spec)):
        ax.plot(wave, spec[i,:], label = labels[i])
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')
    ax.legend(loc = 'best')
    fig.tight_layout()
    if file:
        plt.savefig(file)
    if show:
        plt.show()

def plot_spec_multi(spec, wl = None, labels = None, figsize = (15, 10), file = None, show = True):
    ''' Plot multiple spectra in a facet layout '''
    n = len(spec)
    wave = wl if wl is not None else list(range(spec.shape[1]))
    if not labels:
        labels = ['Spectrum {}'.format(i+1) for i in range(n)]
    n_row = int(np.ceil(n / 2))
    fig, ax = plt.subplots(n_row, 2, figsize = figsize)
    for i in range(n):
        ax[i // 2, i % 2].plot(wave, spec[i,:])
        ax[i // 2, i % 2].set_title(labels[i])
    fig.tight_layout()
    if file:
        plt.savefig(file)
    if show:
        plt.show()

def plot_grouped_bar_chart(nested_vals, labels, group_names, ylab = None, width = 0.4, 
                           title = None, file = None, show = True):
    '''
    Plots a grouped bar chart with 2 or 3 groups where each group gets a color and 
    values from each group corresponding to a particular category are plotted together 
    side-by-side (with categories separated from each other along the X axis).
    Expects a nested list of lists, where each inner list contains the values 
    for a particular group. Labels is a list with the categories.
    '''
    x = np.arange(len(labels))
    n_groups = len(group_names)
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 7))
    if ylab:
        ax.set_ylabel(ylab, fontsize = 14)
    if title:
        ax.set_title(title, fontsize = 15)
    ax.set_xticks(x, labels, fontsize = 14)
    
    if n_groups == 2:
        group_1 = nested_vals[0]
        group_2 = nested_vals[1]
        rects1 = ax.bar(x - width/2, group_1, width, label = group_names[0])
        rects2 = ax.bar(x + width/2, group_2, width, label = group_names[1])
        
    elif n_groups == 3:
        group_1 = nested_vals[0]
        group_2 = nested_vals[1]
        group_3 = nested_vals[2]
        rects1 = ax.bar(x - width, group_1, width, label = group_names[0])
        rects2 = ax.bar(x, group_2, width, label = group_names[1])
        rects3 = ax.bar(x + width, group_3, width, label = group_names[2])
        
    ax.legend(loc = 'best')
    fig.tight_layout()
    if file:
        plt.savefig(file)
    if show:
        plt.show()

def normalize(spec, axis = None):
    ''' Scale all values in spectra or hyperspectral image by max value '''
    spec_max = spec.max(axis = axis, keepdims = True) 
    return spec / spec_max

def center(spec, axis = None):
    ''' Center spectra or image '''
    return spec - spec.mean(axis = axis, keepdims = True)

def scale(spec, axis = None):
    ''' Scale spectra or image by standard deviation '''
    return spec / spec.std(axis = axis, keepdims = True)

def min_max_scale(image, axis = None, to_int = False):
    ''' Scale spectral or hyperspectral image values to be between 0 and 1 '''
    image_min = image.min(axis = axis, keepdims = True)
    image_max = image.max(axis = axis, keepdims = True)
    scaled = (image - image_min) / (image_max - image_min)
    if to_int:
        scaled = (scaled * 255).astype(np.uint8)
    return scaled

def downsample(image, scale = 4):
    ''' Downsample an image by the given scale factor '''
    if scale >= 4:
        start_idx = np.random.randint(1, scale - 1)
    else:
        start_idx = 1 
    downsampled_image = image[start_idx::scale, start_idx::scale, :]
    return downsampled_image

def upsample(image, scale = 4, order = 0, channel_axis = -1):
    ''' 
    Upsample an image by the given scale factor and order 
    (0 for nearest neighbor, 3 for bicubic) 
    '''
    seq = [scale] * image.ndim
    if image.ndim == 3:
        seq[channel_axis] = 1
    if image.ndim == 4:  # batch dimension
        seq[0] = 1

    upsampled = scipy.ndimage.zoom(image, tuple(seq), order = order)
    return upsampled

def write_id_file(path):
    ''' Write image IDs out to a CSV for network ingestion '''
    files = os.listdir(path)
    files = sorted([f[:-4] for f in files if f.split('.')[-1] == 'mat'])
    df = pd.DataFrame({'id': files})
    df.to_csv(os.path.join(path, 'Image_IDs.csv'), index = False)

def calc_MSE(source, target, axis = None):
    ''' Mean-squared error for spectra or hyperspectral images '''
    assert np.all(source.shape == target.shape), 'Source and target must have same shape!'
    if axis is not None:
        # get MSE along axis (e.g. for each spectrum/sample), then take mean of those
        MSE = np.mean(np.mean((source - target)**2, axis = axis))
    else:
        MSE = np.mean((source - target)**2)
    return MSE

def calc_MAE(source, target, axis = None):
    ''' Mean absolute error for spectra or hyperspectral images '''
    assert np.all(source.shape == target.shape), 'Source and target must have same shape!'
    if axis is not None:
        # get MAE along axis (e.g. for each spectrum/sample), then take mean of those
        MAE = np.mean(np.mean(np.abs(source - target), axis = axis))
    else:
        MAE = np.mean(np.abs(source - target))
    return MAE

def calc_PSNR(source, target):
    ''' Peak signal-to-noise ratio for comparing two images '''
    assert np.all(source.shape == target.shape), 'Source and target must have same shape!'
    mse = np.mean((source - target)**2)
    psnr = 10 * np.log10(np.max(source) / mse)
    return psnr

def calc_SSIM(source, target):
    ''' Structural similarity for comparing two images '''
    if source.ndim == 4:
        ssim = 0.
        for i in range(source.shape[0]):
            data_range = source[i].max() - target[i].max()
            ssim += sk_ssim(source[i], target[i], data_range = data_range, channel_axis = -1)
        ssim /= source.shape[0]
    else:
        data_range = source.max() - target.max()
        ssim = sk_ssim(source, target, data_range = data_range, channel_axis = -1)
    return ssim
