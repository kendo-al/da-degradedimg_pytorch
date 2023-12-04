# ResNet56 class
# The original code can be found in torchvision.models.resnet.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# The original code is just modified to use CnnBase. 
import torch
import torch.nn as nn
from cnnmodel import cnn_base

cfgs = {
	'ResNet2_20': ['Basic', 3, 3, 3],
	'ResNet2_32': ['Basic', 5, 5, 5],
	'ResNet2_44': ['Basic', 7, 7, 7],
	'ResNet2_56': ['Basic', 9, 9, 9],
	'ResNet2_110': ['Basic', 18, 18, 18],
    'ResNet2_1202': ['Basic', 200, 200, 200],
}

# Please see the following URL.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride = 1, downsample = None, dilation = 1):
		super(BasicBlock, self).__init__()
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace = True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out

# Please see the following URL.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class Bottleneck(nn.Module):
	expansion = 4
	def __init__(self, inplanes, planes, stride = 1, downsample = None, dilation = 1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False,  dilation = dilation)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size = 1, bias = False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride
		self.downsample = downsample

	def forward(self, x):
		identity = x
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out

class CnnResNet2(cnn_base.CnnBase):
    """ Resnet class."""
    def __init__(self, resnet_name, num_of_classes = 10, num_of_features = 512, init_weights = True, is_cifar = True):
        """ Constructor.

        Keyword arguments:
        resnet_name -- ResNet Type seen in cfgs
        num_of_classes -- the number of class
        num_of_features -- the number of features
        init_weights -- the flag (True: initializing weights)
        is_cifar -- the flag (True: using 32x32 images)
        """
        self.resnet_name = resnet_name
        self.num_of_classes = num_of_classes
        self.inplanes = 16
        self.dilation = 1
        self.is_cifar = is_cifar
        super(CnnResNet2, self).__init__(init_weights, num_of_features)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        """ Make layers by using the BasicBlock class or the Bottleneck class.

        Keyword arguments:
        block -- the block class (BasicBlock, Bottleneck)
        planes -- The number of output features
        blocks -- The number of the unit block
        stride -- the stride
        dilate -- the flag (True: dialtaion)
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, block.expansion*planes, kernel_size = 1, stride = stride, bias = False), nn.BatchNorm2d(block.expansion*planes))

        layers = [block(self.inplanes, planes, stride, downsample, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers += [block(self.inplanes, planes, dilation = self.dilation)]

        return layers

    def _get_block(self):
        """ Get the type of block."""
        if cfgs[self.resnet_name][0] == 'Basic':
            return BasicBlock
        else:
            return Bottleneck

    def _make_layers_for_features(self):
        """ Make feature extractor layers."""
        cfg = cfgs[self.resnet_name]
        if self.is_cifar:
            layers = [nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace = True)]
        else:
            layers = [nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True)]

        planes = 16
        layers += self._make_layer(self._get_block(), planes = planes, blocks = cfg[1], stride = 1)
        planes *= 2
        for i in [2,3]:
            layers += self._make_layer(self._get_block(), planes = planes, blocks = cfg[i], stride = 2)
            planes *= 2

        self.inplanes = 16

        return nn.Sequential(*layers)

    def _make_layers_for_task(self):
        """ Make specific task layers."""
        return nn.Sequential(nn.Linear(64 * self._get_block().expansion, self.num_of_classes))
    
    def _make_avgpool(self):
        """ Make average pooling layer."""
        return nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
