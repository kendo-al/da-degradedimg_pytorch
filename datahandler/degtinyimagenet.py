import torch
import torchvision
from torchvision import transforms

from datahandler import tinyimagenet

class DegTINYIMAGENET(tinyimagenet.TINYIMGNET):
    """ Degraded TINY IMAGENET Dataset.
    Please do not include 'ToTensor' in transform and target_taransform.
    Please use is_to_tensor or is_target_to_tensor flags if you want to apply 'ToTensor'.
    """
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False, is_to_tensor = False, is_target_to_tensor = False, is_normalized = False, deg_transform = None, validation = False):
        """ Constructor.

        Keyword arguments:
        is_to_tensor -- the flag to transform the input to Tensor
        is_target_to_tensor -- the flag to transform the target to Tensor
        is_normalized -- the flag of the normalization
        deg_transform -- taransform for degradation
        validation -- the flag to use the validation data
        """
        super().__init__(root, train, transform, target_transform, validation)
        self.is_to_tensor = is_to_tensor
        self.is_normalized = is_normalized
        self.deg_transform = deg_transform

        if is_to_tensor:
            if self.is_normalized:
                # mean [0.48023695 0.44806705 0.39750365], std [0.27643643 0.26886328 0.28158993]
                normalize = transforms.Normalize(mean = [0.48023695, 0.44806705, 0.39750365], std = [0.27643643, 0.26886328, 0.28158993])
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor(), normalize])
            else:
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor()])
        else:
            self.deg_to_tensor = None
        
        self.is_target_to_tensor = is_target_to_tensor

        transform_str = transform.__repr__()
        target_transform_str = target_transform.__repr__()

        if ('ToTensor' in transform_str) or ('ToTensor' in target_transform_str):
            msg = "Both transform and target_transform should not iclude ToTensor. If you iclude tensor, please use is_to_tensor and is_target_to_tensor."
            raise TypeError(msg)

    def __getitem__(self, index):
        """ Degradation & tensor are applied."""
        clean_img, target = super().__getitem__(index)
        if self.deg_transform is None:
            deg_img = clean_img
        else:
            deg_img = self.deg_transform(clean_img)

        if self.is_to_tensor:
            tensor_deg_img = self.deg_to_tensor(deg_img)
            imgs = tensor_deg_img
        else:
            imgs = deg_img

        if self.is_target_to_tensor:
            tensor_target = torch.tensor(target)
            targets = tensor_target
        else:
            targets = target

        return imgs, targets

    def getCHW(self):
        """ Get channel, height, and width."""
        return (3,64,64)
