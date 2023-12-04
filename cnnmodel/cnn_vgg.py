# VGG class
# The original code can be found in torchvision.models.resnet.
# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
# The original code is just modified to use CnnBase. 
import torch
import torch.nn as nn
from cnnmodel import cnn_base

cfgs = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class CnnVgg(cnn_base.CnnBase):
	"""	VGG class. (32 x 32 sized images are expected.)"""
	def __init__(self, vgg_name, num_of_classes = 10, num_of_features = 512, init_weights = True, is_cifar = True):
		""" Constructor.

		Keyword arguments:
		vgg_name -- the type (seen in cfgs) 
		num_of_classes -- the number of class (default 10)
		num_of_features -- the number of features (default 512)
		init_weights -- the flag of initializing weights
		is_cifar -- the flag (True: use CIFAR)
		"""
		self.vgg_name = vgg_name
		self.num_of_classes = num_of_classes
		self.is_cifar = is_cifar
		super(CnnVgg, self).__init__(init_weights, num_of_features)

	def _make_layers_for_features(self):
		"""	Make feature extractor layers."""
		layers = []
		in_channels = 3
		cfg = cfgs[self.vgg_name]
		cfg.pop()

		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
							nn.BatchNorm2d(x),
							nn.ReLU(inplace=True)]
				in_channels = x
		return nn.Sequential(*layers)
	
	def _make_avgpool(self):
		"""	Make average pooling layer. (max pooling for cifar)"""
		if self.is_cifar:
			return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
		else:
			return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.AdaptiveAvgPool2d((7, 7)))

	def _make_layers_for_task(self):
		"""	Make specific task layers."""
		if self.is_cifar:
			return nn.Sequential(nn.Linear(self.num_of_features, self.num_of_classes))
		else:
			return nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
			nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, self.num_of_classes))
