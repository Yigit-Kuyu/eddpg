import torch
import numpy as np
from typing import Tuple, Optional, Union, List




def to_tensor(data: np.ndarray) -> torch.Tensor:

    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
        
    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
   
    return torch.view_as_complex(data).numpy()


def center_crop(data: torch.Tensor, shape: Tuple[int, int]):
   
    assert 0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1], "Invalid crop shape"
    
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]):

    assert 0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2], "Invalid crop shape"
    
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to, :]

def center_crop_mask(mask: np.ndarray, shape: Tuple[int, int]):

    assert mask.shape[-1] >= shape[1], "Mask width must be larger than or equal to crop width"

    h_from = (mask.shape[-1] - shape[1]) // 2
    h_to = h_from + shape[1]
    return mask[..., h_from:h_to]


def scale(x: torch.Tensor) -> torch.Tensor:
    
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-9)
    
    return x.clamp(0, 1)

    
def complex_abs(data: torch.Tensor) -> torch.Tensor:
   
    assert data.shape[-1] == 2, "Tensor does not have separate complex dim."
    
    return (data**2).sum(dim=-1).sqrt()


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
   
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
   
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    
    if dim is None:
       
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
   
    if dim is None:
        
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def fft2c(data, norm="ortho"):
 
    assert data.shape[-1] == 2, "The last dimension should have a size of 2 corresponding to the real and imaginary parts"
    
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
            torch.fft.fftn(
                torch.view_as_complex(data), dim=(-2, -1), norm=norm)
    )
    data = fftshift(data, dim=[-3, -2])
    
    return data


def ifft2c(data, norm="ortho"):
    
    assert data.shape[-1] == 2, "The last dimension should have a size of 2 corresponding to the real and imaginary parts"
    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
            torch.fft.ifftn(
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])
    
    return data


def apply_mask(data, mask_func, seed=None, args=None): 
    
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:]) 
    mask, num_low_freqs = mask_func(shape, seed=seed, args=args) 
    masked_data = data * mask 
    
    return masked_data, mask, num_low_freqs