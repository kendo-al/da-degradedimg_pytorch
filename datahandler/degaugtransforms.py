import torch
import numpy as np
import copy
import random
from PIL import Image
import datahandler.degtransforms as dtf

def cutdegradation(clean, deg_images, cutratio_min=0.125, cutratio_max=0.75, is_clean_bkg = False, is_clean_cutout = False, applied_prob = 2.0):
    """ Cut Degradation (proposed method).

    Keyword arguments:
    clean -- the clean PIL image
    deg_images -- the list included degraded PIL images
    cutratio_min -- the minimum ratio (default 0.125)
    cutratio_max -- the maximum ratio (default 0.75)
    is_clean_bkg -- the flag to use the clean image as a backgroud (default False)
    is_clean_cutout -- the flag of applying cutout to the clean image (default False)
    applied_prob -- decision probability (default 2.0)
    """
    if len(deg_images)>0:
        img_list = [clean] + deg_images
    else:
        return clean
    
    clean_np = np.array(clean)
    
    h, w, c = clean_np.shape
    for deg_img in deg_images:
        deg_img_np = np.array(deg_img)
        h_d, w_d, c_d = deg_img_np.shape
        if h!=h_d or w!=w_d or c!=c_d:
            raise ValueError("clean and image in deg_images should be the same shape.")
    
    if not(is_clean_bkg):
        random.shuffle(img_list)

    aug_img_np = copy.deepcopy(np.array(img_list[0]))

    if applied_prob<1.0:
        prob = random.uniform(0.0,1.0)
        if prob > applied_prob:
            return Image.fromarray( np.uint8(aug_img_np) )
        
    if (np.array_equal(clean_np, aug_img_np) and is_clean_cutout):
        cut_img_np = np.zeros(clean_np.shape)
        cutratio_h = random.uniform(cutratio_min, cutratio_max)
        cutratio_w = random.uniform(cutratio_min, cutratio_max)
        cut_h = np.int32(h*cutratio_h)
        cut_w = np.int32(w*cutratio_w)
        cut_x = random.randint(0, w-cut_w)
        cut_y = random.randint(0, h-cut_h)
        aug_img_np[cut_y:cut_y+cut_h,cut_x:cut_x+cut_w,:]=cut_img_np[cut_y:cut_y+cut_h,cut_x:cut_x+cut_w,:]
    else:
        for cut_img in img_list[1:]:
            cutratio_h = random.uniform(cutratio_min, cutratio_max)
            cutratio_w = random.uniform(cutratio_min, cutratio_max)
            cut_h = np.int32(h*cutratio_h)
            cut_w = np.int32(w*cutratio_w)
            cut_x = random.randint(0, w-cut_w)
            cut_y = random.randint(0, h-cut_h)
            cut_img_np = np.array(cut_img)
            aug_img_np[cut_y:cut_y+cut_h,cut_x:cut_x+cut_w,:]=cut_img_np[cut_y:cut_y+cut_h,cut_x:cut_x+cut_w,:]

    return Image.fromarray( np.uint8(aug_img_np) )

class CutDegradationApply(torch.nn.Module):
    """ Data augmentation of degraded images with attention to clean images."""
    def __init__(self, deg_type, deg_ranges, deg_lists, cut_ratios, is_clean_bkg, is_clean_cutout = False, applied_prob = 2.0):
        """ Constructor.

		Keyword arguments:
		deg_type -- the type of degradtion
		deg_ranges -- the range of degradation levels
		deg_lists -- the list of degradation levels
        cut_ratio -- the range 0f cut ratio
        is_clean_bkg -- the flag to use the clean image as a backgroud
        is_clean_cutout -- the flag of applying cutout to the clean image (default False)
        applied_prob -- decision probability (default 2.0)
		"""
        super().__init__()
        self.deg_type = deg_type
        self.cut_ratios = cut_ratios
        self.is_clean_bkg = is_clean_bkg
        self.is_clean_cutout = is_clean_cutout
        self.applied_prob = applied_prob
        if deg_ranges is None and deg_lists is None:
            msg = 'deg_ranges or deg_lists should not be None.'
            raise TypeError(msg)
        elif (deg_ranges is not None) and (deg_lists is not None):
            msg = 'Do not input deg_ranges and deg_lists simulteneously.'
            raise TypeError(msg)
        else:
            self.deg_ranges = deg_ranges
            self.deg_lists = deg_lists

        self.deg_func, self.deg_adj = dtf.degradation_function(deg_type)

    def forward(self, img):
        """	Get an augmented image.

        Keyword arguments:
        img : the clean PIL image
		"""
        deg_imgs = []

        if self.deg_ranges is not None:
            for deg_range in self.deg_ranges:
                deg_lev = random.randint(deg_range[0], deg_range[1])
                if self.deg_adj > 1.0:
                    deg_lev = deg_lev / self.deg_adj
                deg_imgs += [self.deg_func(img, deg_lev)]
        else:
            for deg_list in self.deg_lists:
                deg_lev = random.choice(deg_list)
                if self.deg_adj > 1.0:
                    deg_lev = deg_lev / self.deg_adj
                deg_imgs += [self.deg_func(img, deg_lev)]

        return cutdegradation(img, deg_imgs, self.cut_ratios[0], self.cut_ratios[1], self.is_clean_bkg, self.is_clean_cutout, self.applied_prob)

    def __repr__(self):
        return self.__class__.__name__ + '(deg_type={}, deg_range=({},{}))'.format(self.deg_type, self.deg_range[0], self.deg_range[1])

    