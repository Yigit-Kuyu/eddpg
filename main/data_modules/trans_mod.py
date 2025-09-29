import numpy as np
import torch
from typing import NamedTuple
from utils.transform_utils import to_tensor, complex_center_crop, fft2c, ifft2c, apply_mask, complex_abs, scale



class DataTransform:
    def __init__(self, mask_func, resolution, use_seed=True, args=None): 
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed 
        self.args = args 

    def __call__(self, kspace, mask, target, attrs, fname, dataslice, label):
    
        tensor_kspace = to_tensor(kspace)  
        tensor_image = ifft2c(tensor_kspace) 
        crop_size = [self.resolution, self.resolution] 
        cropped_clean_image = complex_center_crop(tensor_image, crop_size) 
        target_tensor_kspace = fft2c(cropped_clean_image) 
        if mask is None and self.mask_func:  
            masked_kspace, mask_, _ = apply_mask(target_tensor_kspace, self.mask_func, seed=self.args.seed, args=self.args)
        else: 
            assert mask is not None, "Mask must be provided if mask_func is not used"
        target_gt = ifft2c(target_tensor_kspace)
        slice_info = {'slice': dataslice, 'label': label, 'original_column_len': tensor_kspace.size(2)}
        return target_tensor_kspace, masked_kspace, mask_, target_gt, fname, slice_info

class U_Sample(NamedTuple):
    image: torch.Tensor
    fname: str
    slice_num: int
    label: int
    metadata: dict    

class DataTransform_Resnet:
    def __init__(self,  mode: str, use_seed: bool = True) -> None:
        
        self.use_seed = use_seed
        self.mode = mode

    def __call__(self, kspace, mask, target, attrs,  fname, slice_num, label):
      
        tensor_kspace = to_tensor(kspace)  
        tensor_image = ifft2c(tensor_kspace)  
        crop_size = [320, 320]
        cropped_clean_image = complex_center_crop(tensor_image, crop_size)  
        target_tensor_kspace = fft2c(cropped_clean_image)  
        input_image = complex_abs(ifft2c(target_tensor_kspace)) 
        input_image = torch.sqrt(torch.sum(input_image ** 2, dim=0))  
        input_image=scale(input_image) 
        
        ret = U_Sample(
           image=input_image, 
           fname=fname,
           slice_num=slice_num, 
           label=label,
           metadata=attrs
        )

        return ret