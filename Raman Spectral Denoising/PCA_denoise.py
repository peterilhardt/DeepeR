import numpy as np
from sklearn.decomposition import PCA

def calc_MSE(source, target):
    if source.ndim == 1:
        source = source.reshape(1, len(source))
        target = target.reshape(1, len(target))
    MSE = np.mean((source - target)**2, axis = 1)
    return np.mean(MSE)

def pca_denoise(spec, n_components = 10):
    pca = PCA(n_components = n_components).fit(spec)
    components = pca.transform(spec)
    return pca.inverse_transform(components)

def get_optimal_pca(source, target, max_components = 100):
    ''' 
    Calculate number of PCs that leads to smallest MSE for PCA denoising. 
    Return that MSE along with the optimal number of components.
    '''
    pca_mse = []
    for i in range(max_components):
        denoised = pca_denoise(source, n_components = i+1)
        MSE = calc_MSE(denoised, target)
        pca_mse.append(MSE)
    min_err_idx = np.argmin(pca_mse)
    return pca_mse[min_err_idx], min_err_idx + 1
