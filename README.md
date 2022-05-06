# High-throughput molecular imaging via deep learning enabled Raman spectroscopy
This repository is the result of a reproducibility study of an original investigation: *High-throughput molecular imaging via deep learning enabled Raman spectroscopy* by Conor C. Horgan, Magnus Jensen, Anika Nagelkerke, Jean-Phillipe St-Pierre, Tom Vercauteren, Molly M. Stevens, and Mads S. Bergholt. The original paper can be found [here](https://pubs.acs.org/doi/10.1021/acs.analchem.1c02178), and much of the code for this work was derived or gathered from the original authors' repository available [here](https://github.com/conor-horgan/DeepeR). 

Citation for the original study:

Conor C. Horgan, Magnus Jensen, Anika Nagelkerke, Jean-Phillipe St-Pierre, Tom Vercauteren, Molly M. Stevens, and Mads S. Bergholt, "High-throughput molecular imaging via deep learning enabled Raman spectroscopy", *Analytical Chemistry* **2021** *93*(48), 15850-15860. DOI: 10.1021/acs.analchem.1c02178.

In addition, a U-Net segmentation model training pipeline was added based on *spectrai*, a separate tool developed by the authors for deep learning-based analysis of hyperspectral data. The original code for *spectrai* was made available [here](https://github.com/conor-horgan/spectrai) and the paper can be found [here](https://arxiv.org/abs/2108.07595). 

Citation for *spectrai*:

Conor C. Horgan and Mads S. Bergholt, "spectrai: A deep learning framework for spectral data", *arXiv preprint* **2021**, arXiv:2108.07595. DOI: 10.48550/arXiv.2108.07595.

Below are instructions for performing various tasks, including training and evaluating models, visualizing results, and applying the model(s) to new data.

## Data and Models
Datasets used in the original work are available from links provided in the [authors' repository](https://github.com/conor-horgan/DeepeR#dataset). Also included are links to pre-trained spectral denoising and hyperspectral super-resolution models provided by the authors. Segmentation masks for the cell dataset were generated for this work and are available in `Segmentation/masks.zip`. 

## Dependencies
Model dependencies can be installed using the provided `environment.yml` file (note that this creates a new Conda environment named "DeepeR"):

```bash
conda env create -f environment.yml
conda activate DeepeR
```

## Training
As described [by the authors](https://github.com/conor-horgan/DeepeR#training), training a new model can be accomplished by downloading the data and running the `train.py` script from the corresponding directory. For example:

```bash
cd "Raman Spectral Denoising"
python train.py --features /path/to/training_inputs.mat --labels /path/to/training_outputs.mat --id my_new_model --epochs 500 --scheduler one-cycle-lr --seed 45 --batch-norm
```

for spectral denoising, or:

```bash
cd "Hyperspectral Super-Resolution"
python train.py --dataset /path/to/training_data_ids.csv --id my_new_model --epochs 600 --lr-image-size 16 --hr-image-size 64 --seed 45
```

for hyperspectral super-resolution, or: 

```bash
cd "Segmentation"
python train.py --dataset /path/to/training_data_ids.csv --labels /path/to/training_masks_directory --dims 2 --classes 2 --id my_new_model --epochs 200 --image-size 64 --normalization None --seed 45
```

for segmentation, where `/path/to/data` indicates an appropriate path. Utility wrappers are also provided here for the denoising and super-resolution models. For example, to train and evaluate a spectral denoising model using an 11-fold cross-validation procedure similar to that used in the original study, edit the parameters in lines 9-32 of `Raman Spectral Denoising/crossval.py`, save, and run:

```bash
cd "Raman Spectral Denoising"
python crossval.py
```

To train and evaluate a hyperspectral super-resolution model (including splitting the data into training and test sets and writing out image ID .csv files like those expected by the data loader), edit the parameters in lines 10-37 of `Hyperspectral Super-Resolution/full_run.py`, save, and run:

```bash
cd "Hyperspectral Super-Resolution"
python full_run.py
```

## Evaluation
As described [by the authors](https://github.com/conor-horgan/DeepeR#testing), evaluating a trained model can be accomplished by downloading the data and model weights (or training a new model) and running the `test.py` script from the corresponding directory. For example:

```bash
cd "Raman Spectral Denoising"
python test.py --features /path/to/test_inputs.mat --labels /path/to/test_outputs.mat --model /path/to/model.pt --seed 45
```

for spectral denoising, or:

```bash
cd "Hyperspectral Super-Resolution"
python test.py --dataset /path/to/test_data_ids.csv --model /path/to/model.pt --lr-image-size 16 --hr-image-size 64 --seed 45
```

for hyperspectral super-resolution, or: 

```bash
cd "Segmentation"
python test.py --dataset /path/to/test_data_ids.csv --labels /path/to/test_masks_directory --model /path/to/model.pt --dims 2 --classes 2 --image-size 64 --normalization None --seed 45
```

for segmentation, where `/path/to/data` or `/path/to/model` indicates an appropriate path to a dataset or model, respectively. To load and test a spectral denoising model (and baseline methods) on a single spectrum or collection of spectra and inspect/visualize the results, edit the paths and parameters in lines 12-13 and 39-46 of `Raman Spectral Denoising/visualize_results.py`, save, and run the file:

```bash
cd "Raman Spectral Denoising"
python visualize_results.py
```

A method is also provided to extract endmember spectra from a hyperspectral image using vertex component analysis (VCA; similar to the original study) and visualize the results. This procedure is found in `Hyperspectral Super-Resolution/VCA.py`. To use, edit the paths and parameters in lines 14-15 and 216-225 of that file, save, and run the file:

```bash
cd "Hyperspectral Super-Resolution"
python VCA.py
```

This file also includes methods for loading and inspecting/visualizing the results of a hyperspectral super-resolution model on a single image, as well as comparing model results to those of baseline methods. 

To load and test a segmentation model on a hyperspectral image and inspect/visualize the results, edit the paths in lines 12-13 and 60 of `Segmentation/visualize_results.py`, save, and run the file:

```bash
cd "Segmentation"
python visualize_results.py
```

## Results
Below shows the results (reported as mean-squared error or MSE) from applying the authors' provided spectral denoising model (`ResUNet.pt`) and a newly-trained model on the test dataset provided (`Test_Inputs.mat` and `Test_Outputs.mat`). Also included are the results from baseline methods as well as those reported in the original study for comparison, though reported results utilized different testing procedures (e.g. cross-validation) and methods for PCA/wavelet denoising.

| | Newly-Trained Model | Authors' Model | Reported |
| --- | --- | --- | --- |
| **ResUNet** | 0.00205 | 0.00195 | 0.00285 |
| **Savitzky-Golay** | 0.0277 | 0.0277 | ~0.0550 | 
| **PCA** | 0.0196 | 0.0196 | 0.0296 |
| **Wavelet** | 0.0207 | 0.0207 | 0.0475 |

The following tables show the results from applying the authors' provided hyperspectral super-resolution models (`RCAN_2x.pt`, `RCAN_3x.pt`, and `RCAN_4x.pt`) as well as newly-trained models on a random test dataset selected from the images provided (see `Hyperspectral Super-Resolution/full_run.py`). Results are shown for all three resolution scales (2x, 3x, and 4x) and the three metrics (MSE, structural similarity (SSIM), and peak signal-to-noise ratio (PSNR)) are shown in different tables. Once again, results from baseline methods and the original study's reported results are included for comparison, though the reported results were generated from a different test set. 

### 2x
| | Newly-Trained Model | Authors' Model | Reported |
| --- | --- | --- | --- |
| **RCAN (MSE)** | 0.0000963 | 0.0000952 | 0.0000780 |
| **Bicubic (MSE)** | 0.000147 | 0.000147 | 0.000114 |
| **Nearest Neighbor (MSE)** | 0.000161 | 0.000161 | 0.000129 |
| **RCAN (SSIM)** | 0.840 | 0.821 | 0.845 |
| **Bicubic (SSIM)** | 0.683 | 0.623 | 0.642 |
| **Nearest Neighbor (SSIM)** | 0.655 | 0.586 | 0.635 |
| **RCAN (PSNR)** | 39.218 | 39.008 | 41.990 |
| **Bicubic (PSNR)** | 39.882 | 39.503 | 40.390 |
| **Nearest Neighbor (PSNR)** | 39.591 | 38.942 | 39.900 |

### 3x
| | Newly-Trained Model | Authors' Model | Reported |
| --- | --- | --- | --- |
| **RCAN (MSE)** | 0.000143 | 0.000143 | 0.000110 |
| **Bicubic (MSE)** | 0.000187 | 0.000187 | 0.000150 |
| **Nearest Neighbor (MSE)** | 0.000247 | 0.000247 | 0.000201 |
| **RCAN (SSIM)** | 0.838 | 0.817 | 0.842 |
| **Bicubic (SSIM)** | 0.644 | 0.574 | 0.631 |
| **Nearest Neighbor (SSIM)** | 0.627 | 0.562 | 0.626 |
| **RCAN (PSNR)** | 37.137 | 36.819 | 40.610 |
| **Bicubic (PSNR)** | 38.149 | 38.123 | 39.280 |
| **Nearest Neighbor (PSNR)** | 36.884 | 36.734 | 38.130 |

### 4x
| | Newly-Trained Model | Authors' Model | Reported |
| --- | --- | --- | --- |
| **RCAN (MSE)** | 0.000202 | 0.000202 | 0.000159 |
| **Bicubic (MSE)** | 0.000258 | 0.000260 | 0.000239 |
| **Nearest Neighbor (MSE)** | 0.000325 | 0.000339 | 0.000309 |
| **RCAN (SSIM)** | 0.847 | 0.800 | 0.828 |
| **Bicubic (SSIM)** | 0.650 | 0.483 | 0.549 |
| **Nearest Neighbor (SSIM)** | 0.632 | 0.448 | 0.526 |
| **RCAN (PSNR)** | 34.696 | 34.688 | 39.270 |
| **Bicubic (PSNR)** | 36.461 | 36.679 | 37.590 |
| **Nearest Neighbor (PSNR)** | 35.308 | 35.600 | 36.570 |

Training a new U-Net segmentation model for 200 epochs resulted in an average cross-entropy loss of 0.343 on the test dataset. 

## Acknowledgements
Thank you to Horgan *et al.* for sharing their work, including code, data, and models. Their studies can be accessed and cited from [here](https://pubs.acs.org/doi/10.1021/acs.analchem.1c02178) and [here](https://arxiv.org/abs/2108.07595). 
