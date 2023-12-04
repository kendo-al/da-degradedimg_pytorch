import os
from torch.utils.data import Dataset
from torchvision import datasets

class TINYIMGNET(datasets.ImageFolder):
    """ TINY ImageNet.
    Before using this class, you have to save the data into proper directories.
    """
    base_dir = 'tiny-imagenet-200'
    train_dir = 'train'
    val_dir = 'val/images'
    def __init__(self, root, train = True, transform = None, target_transform = None, validation = False):
        if train:
            root_dir = os.path.join(root, self.base_dir, self.train_dir)
        elif validation:
            root_dir = os.path.join(root, self.base_dir, self.val_dir)
        else:
            msg = "Test data is not found."
            raise RuntimeError(msg)
        
        super().__init__(root = root_dir, transform = transform, target_transform = target_transform)
