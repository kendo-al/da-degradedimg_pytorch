# ShakeDrop PyramidNet for CIFAR-10/100 class
# The original code can be found in torchvision.models.resnet.
# https://github.com/owruby/shake-drop_pytorch/blob/master/models/shake_pyramidnet.py
# The original code is just modified to use CnnBase. 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from cnnmodel.shakedrop import ShakeDrop
from cnnmodel import cnn_base

cfgs = {
	'Shake110-270': [110, 270],
	'Shake272-200': [272, 200],
}

# Please see the following URL.
# https://github.com/owruby/shake-drop_pytorch/blob/master/models/shake_pyramidnet.py.
class ShakeBasicBlock(nn.Module):
	def __init__(self, in_ch, out_ch, stride=1, p_shakedrop=1.0):
		super(ShakeBasicBlock, self).__init__()
		self.downsampled = stride == 2
		self.branch = self._make_branch(in_ch, out_ch, stride=stride)
		self.shortcut = not self.downsampled and None or nn.AvgPool2d(2)
		self.shake_drop = ShakeDrop(p_shakedrop)

	def forward(self, x):
		h = self.branch(x)
		h = self.shake_drop(h)
		h0 = x if not self.downsampled else self.shortcut(x)
		pad_zero = Variable(torch.zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)).float()).cuda()
		h0 = torch.cat([h0, pad_zero], dim=1)

		return h + h0

	def _make_branch(self, in_ch, out_ch, stride=1):
		return nn.Sequential(
			nn.BatchNorm2d(in_ch),
			nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
			nn.BatchNorm2d(out_ch))

# Please see the following URL.
# https://github.com/owruby/shake-drop_pytorch/blob/master/models/shake_pyramidnet.py.
class CnnShakePyramid(cnn_base.CnnBase):
    """ Shake pyramidnet for CIFAR-10/100.
    32 x 32 sized images are expected.
    """
    def __init__(self, shakepyramid_name, num_of_classes = 10, init_weights = True):
        """ Constructor.

        Keyword arguments:
        shakepyramid_name -- the type of Shake-PyramidNet
        num_of_classes -- the number of class
        init_weights -- : True if you initialize weights.
        """
        self.shakepyramid_name = shakepyramid_name
        self.num_of_classes = num_of_classes
        depth = int(cfgs[self.shakepyramid_name][0])
        alpha = int(cfgs[self.shakepyramid_name][1])
        in_ch = 16
        # for BasicBlock
        self.n_units = (depth - 2) // 6
        in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * self.n_units)) * (i + 1)) for i in range(3 * self.n_units)]
        self.in_chs, self.u_idx = in_chs, 0
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * self.n_units)) * (i + 1)) for i in range(3 * self.n_units)]
        num_of_features = in_chs[-1]

        super(CnnShakePyramid, self).__init__(init_weights, num_of_features)

    def _make_layer(self, n_units, block, stride = 1):
        """ Make layers by using the BasicBlock class or the Bottleneck class.

        Keyword arguments:
        n_units -- units
        block -- block class (ShakeBasicBlock)
        stride -- the size of stride
        """
        layers = []
        for i in range(int(n_units)):
            layers += [block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1], stride, self.ps_shakedrop[self.u_idx])]
            self.u_idx, stride = self.u_idx + 1, 1
        return layers

    def _make_layers_for_features(self):
        """ Make feature extractor layers."""
        layers = [nn.Conv2d(3, self.in_chs[0], 3, padding=1), nn.BatchNorm2d(self.in_chs[0])]
        layers += self._make_layer(self.n_units, ShakeBasicBlock, 1)
        layers += self._make_layer(self.n_units, ShakeBasicBlock, 2)
        layers += self._make_layer(self.n_units, ShakeBasicBlock, 2)
        layers += [nn.BatchNorm2d(self.in_chs[-1]), nn.ReLU(inplace = True)]
        self.u_idx = 0
        return nn.Sequential(*layers)

    def _make_layers_for_task(self):
        """ Make specific task layers."""
        return nn.Sequential(nn.Linear(self.num_of_features, self.num_of_classes))

    def _make_avgpool(self):
        """ Make average pooling layer."""
        return nn.Sequential(nn.AvgPool2d(8))
