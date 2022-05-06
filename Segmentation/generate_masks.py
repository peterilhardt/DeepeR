import os, sys
import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.morphology import opening, closing, disk
from scipy.ndimage import binary_fill_holes

sys.path.append('../Hyperspectral Super-Resolution')
from VCA import vca, non_neg_least_sq

sys.path.append('..')
from helpers import load_image, normalize, plot_image, load_mask 

DATA_BASE_PATH = r'../../data/hyperspectral_super_resolution_data'

def make_mask(img, num_comp):
    A, pure_idx, Y = vca(img, num_comp)
    img_comp = non_neg_least_sq(img, A)
    mask = np.argmax(img_comp, axis = -1)
    return mask

def plot_mask(mask, **kwargs):
    DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                      'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
    colors = label2rgb(mask, colors = DEFAULT_COLORS)
    plot_image(colors, cmap = None, **kwargs)

def update_mask(mask, bkgd_labs, fill_holes = True, erode = True):
    if isinstance(bkgd_labs, int):
        bkgd_labs = [bkgd_labs]

    out_mask = np.zeros_like(mask)
    cell_mask = ~np.isin(mask, bkgd_labs)
    if fill_holes:
        cell_mask = binary_fill_holes(cell_mask)
    if erode:
        footprint = disk(3)
        cell_mask = opening(cell_mask, footprint)
    out_mask[cell_mask] = 1
    return out_mask.astype(np.uint8)

def save_to_png(path, filename, mask):
    im = Image.fromarray(mask)
    im.save(os.path.join(path, filename + '.png'))


if __name__ == '__main__':
    file = 'Cell-HR_Norm_500-01-117-001'
    img = load_image(DATA_BASE_PATH, file)
    img = normalize(img)
    
    plot_image(img)
    for i in range(2, 16):
        mask = make_mask(img, i)
        print('Number of components:', i)
        plot_mask(mask, interpolation = None)
    
    mask = make_mask(img, 6)
    new_mask = update_mask(mask, bkgd_labs = [1,2,4], erode = True)
    plot_mask(new_mask, interpolation = None)

    save_to_png(os.path.join(DATA_BASE_PATH, 'masks'), file, new_mask)
    