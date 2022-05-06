import os, sys
import numpy as np
from skimage.color import label2rgb
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import model

sys.path.append('..')
from helpers import plot_image, load_image, normalize, load_mask

DATA_BASE_PATH = r'../../data/hyperspectral_super_resolution_data'
MODEL_BASE_PATH = r'../../models'


def plot_mask(mask, **kwargs):
    ''' Plots a mask image '''
    DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                      'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
    colors = label2rgb(mask, colors = DEFAULT_COLORS)
    plot_image(colors, cmap = None, **kwargs)

def load_UNet(model_path, dims = 2, spectrum_length = 500, classes = 2, **kwargs):
    ''' Loads a trained UNet segmentation model '''
    assert os.path.exists(model_path), 'Could not find model path!'
    if dims == 2:
        net = model.UNet(dims, channels = spectrum_length, 
                         num_classes = classes, **kwargs).float()
    elif dims == 3:
        net = model.UNet(dims, channels = 1, img_channels = spectrum_length, 
                         num_classes = classes, **kwargs).float()
    else:
        raise ValueError('Unsupported dimensionality for segmentation')

    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_path, map_location = device))
    net.to(device)
    return net

def apply_UNet(model, image):
    ''' Apply loaded UNet segmentation model to image '''
    # Torch expects channels first and a batch dimension
    img = np.moveaxis(image, -1, 0)[np.newaxis, ...]
    img = torch.tensor(img).float().to(torch.device('cpu'))

    model.eval()
    with torch.no_grad():
        out = model(img).cpu().detach()
        out = np.squeeze(out.numpy())
        out = np.moveaxis(out, 0, -1)
    return out

def to_mask(seg_block):
    ''' Convert output of UNet to mask '''
    out = seg_block.argmax(axis = -1).astype(np.uint8)
    return out


if __name__ == '__main__':
    img_file = 'Cell-HR_Norm_500-03-077-134'

    # load and normalize image
    img = load_image(DATA_BASE_PATH, img_file)
    img = normalize(img)
    
    # directory for output images
    if not os.path.exists('./images'):
        os.mkdir('./images')
    plot_image(img, file = './images/{}_original.png'.format(img_file), show = False)

    # load UNet model and apply to image
    unet = load_UNet(os.path.join(MODEL_BASE_PATH, 'UNet_Segmentation.pt'), dims = 2, 
                     classes = 2, normalization = None)
    unet_img = apply_UNet(unet, img)
    unet_mask = to_mask(unet_img)

    # load true mask
    true_mask = load_mask(os.path.join(DATA_BASE_PATH, 'masks'), img_file)

    # generate plots
    plot_mask(true_mask, file = './images/{}_true_mask.png'.format(img_file), show = False)
    plot_mask(unet_mask, file = './images/{}_model_mask.png'.format(img_file), show = False)

    for i in range(unet_img.shape[2]):
        plot_image(unet_img[:,:,i], 
                   file = './images/{}_model_output_class{}.png'.format(img_file, i), 
                   show = False)
