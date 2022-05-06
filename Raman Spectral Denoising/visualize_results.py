import os, sys
import numpy as np
import scipy.signal

sys.path.append('..')
from helpers import load_spectra, plot_spec_overlay, plot_spec_multi, \
    normalize, plot_spec, min_max_scale

import torch
import model, PCA_denoise, wavelet_denoise

DATA_BASE_PATH = r'../../data/spectral_denoising_data'
MODEL_BASE_PATH = r'../../models'


def load_ResUNet(model_path, batch_norm = True):
    ''' Loads a trained ResUNet model '''
    assert os.path.exists(model_path), 'Could not find model path!'
    net = model.ResUNet(3, batch_norm).float()
    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_path, map_location = device))
    net.to(device)
    return net

def apply_ResUNet(model, spectrum):
    ''' Apply loaded ResUNet model to spectrum '''
    # Torch expects an extra dimension and a batch dimension
    if spectrum.ndim == 1:
        spec = spectrum.reshape(1, -1)[np.newaxis, ...]
    else:
        spec_length = spectrum.shape[-1]
        spec = spectrum.reshape(-1, 1, spec_length)
    spec = torch.tensor(spec).float().to(torch.device('cpu'))

    model.eval()
    with torch.no_grad():
        out = model(spec).cpu().detach()
        out = np.squeeze(out.numpy())
    return out


if __name__ == '__main__':
    SPEC_INDEX = 4000

    # load low-SNR and high-SNR spectra and normalize
    inputs = load_spectra(DATA_BASE_PATH, 'Test_Inputs.mat')
    outputs = load_spectra(DATA_BASE_PATH, 'Test_Outputs.mat')
    input = min_max_scale(inputs, axis = -1)[SPEC_INDEX, :]
    output = min_max_scale(outputs, axis = -1)[SPEC_INDEX, :]
    wl = wl = np.arange(600, 1800, (1800 - 600) / len(input))

    # plot the original versions
    if not os.path.exists('./images'):
        os.mkdir('./images')
    file = './images/original_low_snr_spec.png'
    plot_spec(input, wl, file = file, show = False)
    file = './images/original_high_snr_spec.png'
    plot_spec(output, wl, file = file, show = False)

    # load model, apply to spectrum, and plot result
    model_file = os.path.join(MODEL_BASE_PATH, 'ResUNet_trained.pt')
    net = load_ResUNet(model_file, batch_norm = True)
    output_ResUNet = apply_ResUNet(net, input)
    output_ResUNet = min_max_scale(output_ResUNet)
    file = './images/denoised_spec_ResUNet.png'
    plot_spec(output_ResUNet, wl, file = file, show = False)

    # also apply Savitzky-Golay, PCA, and wavelet denoising and plot results
    output_sg = scipy.signal.savgol_filter(input, window_length = 9, 
                                           polyorder = 1)
    output_sg = min_max_scale(output_sg)
    # this returns the optimal number of PCs to use
    _, num_PCs = PCA_denoise.get_optimal_pca(inputs, outputs, 
                                             max_components = 10)
    output_pca = PCA_denoise.pca_denoise(inputs, num_PCs)[SPEC_INDEX, :]
    output_pca = min_max_scale(output_pca)
    # this returns the optimal wavelet type and level to use
    _, w_type, w_level = wavelet_denoise.get_optimal_wavelet(inputs, outputs,
                                                            max_level = 2)
    output_wl = wavelet_denoise.wavelet_denoise(input, w_type, w_level)[0]
    output_wl = min_max_scale(output_wl)

    file = './images/denoised_spec_SG.png'
    plot_spec(output_sg, wl, file = file, show = False)
    file = './images/denoised_spec_PCA.png'
    plot_spec(output_pca, wl, file = file, show = False)
    file = './images/denoised_spec_wavelet.png'
    plot_spec(output_wl, wl, file = file, show = False)

    # show facets and overlay plots for comparison
    all_spec = np.vstack([input, output, output_sg, output_wl, 
                          output_pca, output_ResUNet])
    labels = ['Input', 'Target', 'Savitzky-Golay', 'Wavelet', 'PCA', 'ResUNet']
    file = './images/all_spec_facet.png'
    plot_spec_multi(all_spec, wl, labels = labels, file = file, show = False,
                    figsize = (15, 12))
    file = './images/all_spec_overlay.png'
    plot_spec_overlay(all_spec, wl, labels = labels, file = file, show = False)
