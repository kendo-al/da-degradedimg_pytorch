import torch
import numpy as np
import copy
import random
from PIL import Image
import imagedegrade.np as np_degrade # https://github.com/mastnk/imagedegrade

def cutout(input_img, cut_prob=0.5, cutratio_min=0.125, cutratio_max=0.75, is_gauss = False, sigma = 50):
    """ Cutout & Random erasing.

    Keyword arguments:
    input_img -- the input PIL image
    cut_prob -- decision probability (defalut 0.5)
    cutratio_min -- the minimum ratio (default 0.125) 
    cutratio_max -- the maximum ratio (default 0.75)
    is_gauss -- the flag to apply Gaussin noise (default False)
    sigma -- the 8-bit base standard deviation of Gaussian noise (default 50)
    """
    
    if random.uniform(0.0, 1.0) > cut_prob:
        return input_img
    
    input_np = np.array(input_img)
    
    h, w, c = input_np.shape
    aug_img_np = copy.deepcopy(input_np)
    cut_img_np = np.zeros(input_np.shape)
    if is_gauss:
        cut_img_np = np_degrade.noise(cut_img_np, sigma)

    cutratio_h = random.uniform(cutratio_min, cutratio_max)
    cutratio_w = random.uniform(cutratio_min, cutratio_max)
    cut_h = np.int32(h*cutratio_h)
    cut_w = np.int32(w*cutratio_w)
    cut_x = random.randint(0, w-cut_w)
    cut_y = random.randint(0, h-cut_h)
    aug_img_np[cut_y:cut_y+cut_h,cut_x:cut_x+cut_w,:]=cut_img_np[cut_y:cut_y+cut_h,cut_x:cut_x+cut_w,:]

    return Image.fromarray( np.uint8(aug_img_np) )

class CutOutApply(torch.nn.Module):
    """ Data augmentation of cutout and random erasing."""
    def __init__(self, cut_prob=0.5, cutratio_min=0.125, cutratio_max=0.75, is_gauss = False, sigma = 50):
        """ Contructor.

		Keyword arguments:
        cut_prob -- decision probability (defalut 0.5)
        cutratio_min -- the minimum ratio (default 0.125) 
        cutratio_max -- the maximum ratio (default 0.75)
        is_gauss -- the flag to apply Gaussin noise (default False)
        sigma -- the 8-bit base standard deviation of Gaussian noise (default 50)
		"""
        super().__init__()
        self.cut_prob = cut_prob
        self.cutratio_min = cutratio_min
        self.cutratio_max = cutratio_max
        self.is_gauss = is_gauss
        self.sigma = sigma

    def forward(self, img):
        """ Get a cutout image.

        Keyword arguments:
        img -- the input PIL image
        """
        return cutout(img, self.cut_prob, self.cutratio_min, self.cutratio_max, self.is_gauss, self.sigma)
   