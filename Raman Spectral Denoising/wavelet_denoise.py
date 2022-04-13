import numpy as np
import pywt

def calc_MSE(source, target):
    if source.ndim == 1:
        source = source.reshape(1, len(source))
        target = target.reshape(1, len(target))
    MSE = np.mean((source - target)**2, axis = 1)
    return np.mean(MSE)

def wavelet_denoise(spec, wavelet = 'db4', level = 1, mode = 'symmetric', threshold = 'hard'):
    if spec.ndim == 1:
        spec = spec.reshape(1, len(spec))
    n, spec_len = spec.shape
        
    # mean absolute deviation
    def maddev(x):
        return np.mean(np.absolute(x - np.mean(x, axis = 1).reshape(n, 1)), axis = 1)
    
    # get wavelet coefficients
    coeff = pywt.wavedec(spec, wavelet, mode = mode)
    
    # calculate and apply thresholds
    sigma = (1 / 0.6745) * maddev(coeff[-level])
    uthresh = (sigma * np.sqrt(2 * np.log(spec_len))).reshape(n, 1)
    coeff[1:] = (pywt.threshold(i, value = uthresh, mode = threshold) for i in coeff[1:])
    
    # Reconstruct signal using thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode = mode)

def get_optimal_wavelet(source, target, max_level = 3):
    ''' 
    Calculate wavelet type/level that leads to smallest MSE for wavelet denoising. 
    Return that MSE along with the wavelet type and decomposition level.
    '''
    if source.ndim == 1:
        source = source.reshape(1, len(source))
        target = target.reshape(1, len(target))
    n, spec_len = source.shape
    
    wavelet_mse = []
    wavelet_type = []
    wavelet_level = []
    for wavelet in pywt.wavelist(kind = 'discrete'):
        # calculate max useful level of decomposition for wavelet type
        max_useful_level = pywt.dwt_max_level(spec_len, wavelet)
        max_to_use = min(max_useful_level, max_level) if max_level is not None else max_useful_level
        for lev in range(max_to_use):
            # get reconstructed (denoised) spectra
            denoised = wavelet_denoise(source, wavelet = wavelet, level = lev+1)
            MSE = calc_MSE(denoised, target)
            wavelet_mse.append(MSE)
            wavelet_type.append(wavelet)
            wavelet_level.append(lev + 1)
            
    min_err_idx = np.argmin(wavelet_mse)
    return wavelet_mse[min_err_idx], wavelet_type[min_err_idx], wavelet_level[min_err_idx]
