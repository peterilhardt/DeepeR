import os
import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset

class RamanImageDataset(Dataset):
    def __init__(self, image_ids, path, mask_path = None, batch_size = 2, image_size = 64, 
                 spectrum_len = 500, spectrum_shift = 0., spectrum_flip = False, 
                 horizontal_flip = False, vertical_flip = False, rotate = False, patch = False):
        self.image_ids = image_ids
        self.path = path
        self.mask_path = mask_path if mask_path else path
        self.batch_size = batch_size
        self.image_size = image_size
        self.spectrum_len = spectrum_len
        self.spectrum_shift = spectrum_shift
        self.spectrum_flip = spectrum_flip
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate = rotate
        self.patch = patch
        self.on_epoch_end()
        
    def load_image(self, id_name):
        input_path = os.path.join(self.path, id_name + ".mat")
        output_data = scipy.io.loadmat(input_path)
        output_values = list(output_data.values())
        output_image = output_values[3]
        return output_image

    def load_mask(self, id_name):
        mask_path = os.path.join(self.mask_path, id_name + ".png")
        mask_img = Image.open(mask_path)
        mask = np.asarray(mask_img).astype(np.float64)
        #mask -= 1
        return mask
    
    def pad_image(self, image, size, patch, horiz_frac = 0.5, vert_frac = 0.5):
        if image.shape[0] == size and image.shape[1] == size:
            padded_image = image
        elif image.shape[0] > size and image.shape[1] > size:
            if patch:
                padded_image = self.get_image_patch(image, size, horiz_frac, vert_frac)
            else:
                padded_image = self.center_crop_image(image, size)                    
        else:
            padded_image = image
            if padded_image.shape[0] > size:
                if patch:
                    padded_image = self.get_image_patch(padded_image, size, horiz_frac, vert_frac)
                else:
                    padded_image = self.center_crop_image(padded_image, size) 
            else:           
                pad_before = int(np.floor((size - padded_image.shape[0])/2))
                pad_after = int(np.ceil((size - padded_image.shape[0])/2))
                padded_image = np.pad(padded_image, ((pad_before, pad_after), (0,0), (0, 0)), 'reflect')

            if padded_image.shape[1] > size:
                if patch:
                    padded_image = self.get_image_patch(padded_image, size, horiz_frac, vert_frac)
                else:
                    padded_image = self.center_crop_image(padded_image, size) 
            else:           
                pad_before = int(np.floor((size - padded_image.shape[1])/2))
                pad_after = int(np.ceil((size - padded_image.shape[1])/2))
                padded_image = np.pad(padded_image, ((0,0), (pad_before, pad_after), (0, 0)), 'reflect')

        return padded_image

    def get_image_patch(self, image, patch_size, horiz_frac, vert_frac):                   
        if image.shape[0] > patch_size:
            start_idx_x = int(np.round(vert_frac * (image.shape[0]-patch_size)))
            end_idx_x = start_idx_x + patch_size
        else:
            start_idx_x = 0
            end_idx_x = image.shape[0]

        if image.shape[1] > patch_size:
            start_idx_y = int(np.round(horiz_frac * (image.shape[1]-patch_size)))
            end_idx_y = start_idx_y + patch_size
        else:
            start_idx_y = 0
            end_idx_y = image.shape[1]

        image_patch = image[start_idx_x:end_idx_x,start_idx_y:end_idx_y,:]
        return image_patch

    def center_crop_image(self, image, image_size):
        cropped_image = image
        if image.shape[0] > image_size:
            dif = int(np.floor((image.shape[0] - image_size)/2))
            cropped_image = cropped_image[dif:image_size+dif,:,:]

        if image.shape[1] > image_size:
            dif = int(np.floor((image.shape[1] - image_size)/2))
            cropped_image = cropped_image[:,dif:image_size+dif,:]
        return cropped_image
     
    def flip_axis(self, image, axis):
        image = np.asarray(image).swapaxes(axis, 0)
        image = image[::-1, ...]
        image = image.swapaxes(0, axis)
        return image
        
    def rotate_spectral_image(self, image, rotation_extent = 0.5):
        if rotation_extent < 0.25:
            rotation = 1
        elif rotation_extent < 0.5:
            rotation = 2
        elif rotation_extent < 0.75:
            rotation = 3
        else:
            rotation = 0
        image = np.rot90(image, rotation)
        return image
    
    def shift_spectrum(self, image, shift_range):
        shifted_spectrum_image = image
        spectrum_shift_range = int(np.round(shift_range*image.shape[2]))
        if spectrum_shift_range > 0:
            shifted_spectrum_image = np.pad(image[:,:,spectrum_shift_range:], ((0,0), (0,0), (0,abs(spectrum_shift_range))), 'reflect')
        elif spectrum_shift_range < 0:
            shifted_spectrum_image = np.pad(image[:,:,:spectrum_shift_range], ((0,0), (0,0), (abs(spectrum_shift_range), 0)), 'reflect')
        return shifted_spectrum_image
    
    def spectrum_padding(self, image, spectrum_length):
        if image.shape[-1] == spectrum_length:
            padded_spectrum_image = image
        elif image.shape[-1] > spectrum_length:
            padded_spectrum_image = image[:,:,0:spectrum_length]
        else:
            padded_spectrum_image = np.pad(image, ((0,0), (0,0), (0, spectrum_length - image.shape[-1])), 'reflect')
        return padded_spectrum_image
    
    def normalise_image(self, image):
        image_max = np.tile(np.amax(image),image.shape)
        normalised_image = np.divide(image,image_max)
        return normalised_image 
    
    def __getitem__(self, idx):
        img = self.load_image(self.image_ids[idx])
        mask = self.load_mask(self.image_ids[idx])[..., np.newaxis]

        # --------------- Image Data Augmentations --------------- 
        horiz_frac = np.random.random()
        vert_frac = np.random.random()
        img = self.pad_image(img, self.image_size, self.patch, horiz_frac, vert_frac)
        mask = self.pad_image(mask, self.image_size, self.patch, horiz_frac, vert_frac)

        if self.horizontal_flip: 
            if np.random.random() < 0.5:    
                img = self.flip_axis(img, 1)
                mask = self.flip_axis(mask, 1)

        if self.vertical_flip: 
            if np.random.random() < 0.5:    
                img = self.flip_axis(img, 0)
                mask = self.flip_axis(mask, 0)

        if self.rotate:
            extent = np.random.random()
            img = self.rotate_spectral_image(img, rotation_extent = extent)
            mask = self.rotate_spectral_image(mask, rotation_extent = extent)

        # --------------- Spectral Data Augmentations --------------- 
        if self.spectrum_shift != 0.0:
            shift_range = np.random.uniform(-self.spectrum_shift, self.spectrum_shift)
            img = self.shift_spectrum(img, shift_range)

        img = self.spectrum_padding(img, self.spectrum_len)

        if self.spectrum_flip:    
            if np.random.random() < 0.5:
                img = self.flip_axis(img, 2)

        # --------------- Normalisation --------------- 
        img = self.normalise_image(img)
        img = np.moveaxis(img, -1, 0).copy()
        mask = np.squeeze(mask).copy()

        sample = {'image': img, 'mask': mask}
        return sample
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return len(self.image_ids)
