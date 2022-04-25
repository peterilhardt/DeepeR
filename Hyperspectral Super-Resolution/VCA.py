import os, sys
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import matplotlib

sys.path.append('..')
from helpers import load_image, plot_spec_overlay, plot_spec_multi, \
    normalize, plot_image, downsample, upsample

import torch
import model

DATA_BASE_PATH = r'../../data/hyperspectral_super_resolution_data'
MODEL_BASE_PATH = r'../../models'


def estimate_snr(Y, r_m, x):
    L, N = Y.shape   # (num_channels, num_pixels)
    p, _ = x.shape   # p number of endmembers

    P_y = np.sum(Y**2) / float(N)
    P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
    
    snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))
    return snr_est

def vca(Y, R, snr_input = 0, seed = 42):
    """
    Vertex Component Analysis (adapted from code by Adrien 
    Lagrange: https://github.com/Laadr/VCA)

    ------- Input variables -------------
     Y - array with dimensions L (channels) x N (pixels) where
         each pixel is a linear mixture of R endmembers;
         signatures Y = M x s, where s = gamma x alpha;
         gamma is an illumination perturbation factor and
         alpha are the abundance fractions of each endmember
     R - positive integer number of endmembers in the scene
     snr_input - (float; optional) signal-to-noise ratio (dB)

    ------- Output variables -----------
    Ae     - estimated mixing matrix (endmember signatures)
    idx    - pixels that were chosen to be the most pure
    Yp     - data matrix Y projected   
    """
    
    # Initializations
    cube = False
    if len(Y.shape) == 3:
        cube = True
        nx, ny, nf = Y.shape
        Y = Y.reshape((nx * ny, nf)).T

    L, N = Y.shape   # L number of bands (channels), N number of pixels

    R = int(R)
    if (R <= 0 or R > L):  
        raise ValueError('R must be integer between 1 and L')

    if seed:
        np.random.seed(seed)

    # Compute projection and SNR estimates
    y_m = np.mean(Y, axis = 1, keepdims = True)
    Y_o = Y - y_m    # mean-center
    Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:,:R]  # computes R-projection matrix 
    x_p = np.dot(Ud.T, Y_o)  # projects zero-mean data onto R-subspace
    
    if snr_input == 0:
        SNR = estimate_snr(Y, y_m, x_p)
        print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        print("input SNR = {}[dB]".format(SNR))

    SNR_thresh = 15 + 10 * np.log10(R)

    # Choosing projective projection or projection onto R-1 subspace
    if SNR < SNR_thresh:
        print("Using projection onto R-1")

        d = R - 1
        Ud = Ud[:,:d]
        x = x_p[:d,:]

        Yp = np.dot(Ud, x) + y_m  # back in dimension L
        c = np.amax(np.sum(x**2, axis = 0))**0.5
        y = np.vstack((x, c * np.ones((1,N))))
    else:
        print("Using projective projection")

        d = R
        Ud = np.linalg.svd(np.dot(Y, Y.T) / float(N))[0][:,:d]  # computes R-projection matrix 
        x = np.dot(Ud.T, Y)
        
        Yp = np.dot(Ud, x)  # back in dimension L (note that x_p has non-zero mean)
        u = np.mean(x, axis = 1, keepdims = True)   # equivalent to u = Ud.T * r_m
        y =  x / np.dot(u.T, x)

    # VCA algorithm
    idx = np.zeros((R), dtype = int)
    A = np.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = np.random.rand(R, 1)   
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / np.linalg.norm(f)
        v = np.dot(f.T, y)
        idx[i] = np.argmax(np.absolute(v))
        A[:,i] = y[:,idx[i]]   # same as x[:,idx(i)]

    Ae = Yp[:,idx]
    if cube:
        return Ae.T, idx, Yp.T.reshape((nx, ny, nf))
    return Ae.T, idx, Yp.T

def least_sq(img, comp):
    ''' Least-squares estimation of component abundances in spectral mixture '''
    nx, ny, nf = img.shape
    n_comp = len(comp)
    Y = img.reshape((nx * ny, nf)).T  # (channels, pixels)
    X = comp.T  # (channels, components)
    conc = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))  # (components, pixels)
    return conc.T.reshape((nx, ny, n_comp))

def non_neg_least_sq(img, comp):
    ''' Non-negative least-squares estimation of component abundances in spectral mixture '''
    nx, ny, nf = img.shape
    n_comp = len(comp)
    Y = img.reshape((nx * ny, nf))  # (pixels, channels)
    X = comp.T  # (channels, components)
    conc = np.zeros((nx * ny, n_comp))  # (pixels, components)
    for i in range(len(Y)):  # iterate through each pixel
        spec = Y[i,:]
        c, _ = nnls(X, spec)
        conc[i,:] = c
    return conc.reshape((nx, ny, n_comp))

def plot_VCA_image(img_components, comp_colors, labels = None, colorbar = True, file = None, show = True):
    ''' 
    Takes the per-pixel VCA component least-squares fits and a list of colors/labels and 
    plots an image with the component distributions overlaid, like the kind shown in 
    Horgan et al., 2021 (https://pubs.acs.org/doi/10.1021/acs.analchem.1c02178) 
    '''
    n = len(comp_colors)
    
    # normalize components to sum to 1
    img_comp_scaled = img_components / img_components.sum(axis = 2)[..., np.newaxis]
    #img_comp_scaled = np.clip(img_components, 0, 1)
    
    # convert pixel values to RGB by taking weighted sum of component
    # abundances and their respective colors
    img_comp_rgb = np.dot(img_comp_scaled, np.array(comp_colors))
    img_comp_rgb = img_comp_rgb.round().astype(np.uint8)
    
    # plot RGB image
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.imshow(img_comp_rgb)
    ax.axis('off')
    
    # need to make custom discrete colorbar for component labels
    if colorbar:
        if not labels:
            labels = ['Component {}'.format(i+1) for i in range(n)]
        cmap = matplotlib.colors.ListedColormap(np.array(comp_colors)[::-1,:] / 255.)
        bounds = np.arange(0, n + 1) + 0.5
        norm = matplotlib.colors.BoundaryNorm(bounds, n)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[::-1][norm(x)])
        #cb = matplotlib.colorbar.ColorbarBase(ax, cmap = cmap, norm = norm, format = fmt)
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
        fig.colorbar(sm, boundaries = bounds, format = fmt, ticks = np.arange(0, n + 1))
    
    fig.tight_layout()
    if file:
        plt.savefig(file)
    if show:
        plt.show()

def compute_component_accuracy(model_comp, target_comp):
    '''
    Computes classification accuracy across all pixels for a model-derived
    VCA component distribution (determined by NNLS) as compared to the 
    true component distribution.
    '''
    assert np.all(model_comp.shape == target_comp.shape), 'Inputs should have the same shape!'
    pred = model_comp.argmax(axis = 2)
    true = target_comp.argmax(axis = 2)
    return np.mean(pred == true)

def load_RCAN(model_path, spectrum_length = 500, scale = 2):
    ''' Loads a trained hyperspectral RCAN model '''
    assert os.path.exists(model_path), 'Could not find model path!'
    net = model.Hyperspectral_RCAN(spectrum_length, scale).float()
    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_path, map_location = device))
    net.to(device)
    return net

def apply_RCAN(model, image):
    ''' Apply loaded RCAN model to image '''
    # Torch expects channels first and a batch dimension
    img = np.moveaxis(image, -1, 0)[np.newaxis, ...]
    img = torch.tensor(img).float().to(torch.device('cpu'))

    model.eval()
    with torch.no_grad():
        out = model(img).cpu().detach()
        out = np.squeeze(out.numpy())
        out = np.moveaxis(out, 0, -1)
    return out


if __name__ == '__main__':
    SCALE = 4

    # load high-resolution image and normalize
    img = load_image(DATA_BASE_PATH, 'Cell-HR_Norm_500-03-079-136')
    # img = load_image(DATA_BASE_PATH, 'Cell-HR_Norm_500-03-077-130')
    img = normalize(img)

    # find components using VCA
    A, pure_idx, Y = vca(img, 5)
    # A, pure_idx, Y = vca(img, 4)

    # plot the original image and component spectra
    if not os.path.exists('./images'):
        os.mkdir('./images')
    plot_image(img, file = './images/high_res_image.png', show = False)
    plot_spec_overlay(A, file = './images/VCA_spectra_overlaid.png', show = False)
    plot_spec_multi(A, file = './images/VCA_spectra_facet.png', show = False)

    # fit components to each pixel via non-negative least squares
    img_comp = non_neg_least_sq(img, A)

    # plot a VCA components image
    colors = [(53, 83, 169),
            (0, 0, 0),
            (41, 187, 76),
            (248, 239, 24),
            (247, 28, 36)]
    labels = ['Nucleic Acids', 'Background', 'Proteins', 'Lipids', 'Proteins/Lipids']

    # colors = [(53, 83, 169), 
    #             (0, 0, 0),
    #             (248, 239, 24),
    #             (41, 187, 76)]
    # labels = ['Nucleic Acids', 'Background', 'Lipids', 'Proteins']
    
    plot_VCA_image(img_comp, colors, labels = labels, 
                   file = './images/VCA_components_high_res.png', 
                   show = False)
    
    # downsample image, load RCAN model, and apply model to downsampled image
    img_ds = downsample(img, scale = SCALE)
    image_file = './images/low_res_image_{}x.png'.format(SCALE)
    plot_image(img_ds, file = image_file, show = False)
    model_file = 'RCAN_{}x_trained.pt'.format(SCALE)
    net = load_RCAN(os.path.join(MODEL_BASE_PATH, model_file), scale = SCALE)
    img_rcan = apply_RCAN(net, img_ds)
    image_file = './images/super_res_image_RCAN_{}x.png'.format(SCALE)
    plot_image(img_rcan, file = image_file, show = False)

    # fit VCA components to RCAN super-resolution image via NNLS and visualize
    img_rcan_comp = non_neg_least_sq(img_rcan, A)
    plot_VCA_image(img_rcan_comp, colors, labels = labels, 
                   file = './images/VCA_components_RCAN.png', 
                   show = False)

    # repeat process for nearest-neighbor and bicubic upsampling
    img_nn = upsample(img_ds, scale = SCALE, order = 0)
    img_bc = upsample(img_ds, scale = SCALE, order = 3)
    image_file = './images/super_res_image_nearest_neighbor_{}x.png'.format(SCALE)
    plot_image(img_nn, file = image_file, show = False)
    image_file = './images/super_res_image_bicubic_{}x.png'.format(SCALE)
    plot_image(img_bc, file = image_file, show = False)
    
    img_nn_comp = non_neg_least_sq(img_nn, A)
    img_bc_comp = non_neg_least_sq(img_bc, A)

    plot_VCA_image(img_nn_comp, colors, labels = labels, 
                   file = './images/VCA_components_nearest_neighbor.png', 
                   show = False)
    plot_VCA_image(img_bc_comp, colors, labels = labels, 
                   file = './images/VCA_components_bicubic.png', 
                   show = False)
    
    # compute and print component distribution accuracies across all pixels
    nn_acc = compute_component_accuracy(img_nn_comp, img_comp)
    bc_acc = compute_component_accuracy(img_bc_comp, img_comp)
    rcan_acc = compute_component_accuracy(img_rcan_comp, img_comp)
    print('Nearest neighbor accuracy:', nn_acc.round(3))
    print('Bicubic accuracy:', bc_acc.round(3))
    print('RCAN accuracy:', rcan_acc.round(3))
