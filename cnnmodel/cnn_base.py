# Base class of CNN models
import torch
import torch.nn as nn

class CnnBase(nn.Module):
	"""	Base Class of CNNs."""
	def __init__(self, init_weights = True, num_of_features = 0):
		"""	Constructor.

		Keyword arguments:
		num_of_features -- the number of features
		init_weights -- the flag to initialize weights (default True)
		"""
		super(CnnBase, self).__init__()
		self.num_of_features = num_of_features
		self.features = self._make_layers_for_features()
		self.avgpool = self._make_avgpool()
		self.task = self._make_layers_for_task()
		if init_weights:
			self.features.apply(self._initilize_weights_for_layer)
			self.task.apply(self._initilize_weights_for_layer)

	def train_features(self, is_training):
		"""	Make the features trainable or not.

		Keyword arguments:
		is_training -- the flag (train: True)
		"""
		for param in self.features.parameters():
			param.requires_grad = is_training

	def freeze_features(self, is_training):
		"""	Freeze all layers of features including bactchnorm layers and dropout.

		Keyword arguments:
		is_training -- the flag (train: True)
		"""
		if is_training:
			self.features.train()
		else:
			self.features.eval()

	def train_task(self, is_training):
		"""	Make the task of clean CNN trainable or not.

		Keyword arguments:
		is_training -- the flag (train: True)
		"""
		for param in self.task.parameters():
			param.requires_grad = is_training

	def freeze_task(self, is_training):
		"""	Freeze all layers of task including bactchnorm layers and dropout.

		Keyword arguments:
		is_training -- the flag (train: True)
		"""
		if is_training:
			self.task.train()
		else:
			self.task.eval()

	def forward(self, x_img):
		"""	Forward.

		Keyword arguments:
		x_clean -- input images
		"""
		img_feat = self.features(x_img)
		img_feat = self.avgpool(img_feat)
		img_feat = torch.flatten(img_feat, 1)
		fwd_out = self.task(img_feat)
		return fwd_out

	def _make_layers_for_features(self):
		"""	Make feature extractor layers."""
		msg = "Feature layers have not been implemeted."
		raise NotImplementedError(msg)
		return None
	
	def _make_avgpool(self):
		"""	Make AvgPool layer."""
		msg = "AvgPool layer has not been implemeted."
		raise NotImplementedError(msg)
		return None

	def _make_layers_for_task(self):
		"""	Make specific task layers."""
		msg = "Feature layers have not been implemeted."
		raise NotImplementedError(msg)
		return None

	def _initilize_weights_for_layer(self, layer):
		"""	Initialize each layer depends on the layer type.

		Keyword arguments:
		layer -- the target layer
		"""
		if isinstance(layer, nn.Conv2d):
			nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				nn.init.constant_(layer.bias, 0)
		elif isinstance(layer, nn.BatchNorm2d):
			nn.init.constant_(layer.weight, 1)
			nn.init.constant_(layer.bias, 0)
		elif isinstance(layer, nn.Linear):
			nn.init.normal_(layer.weight, 0, 0.01)
			nn.init.constant_(layer.bias, 0)

	def train_features_partially_startswith(self, num_list, is_training):
		"""	Make the features trainable or not.

		Keyword arguments:
		num_list -- the list of layer numbers in features
		is_training -- the flag (train: True)
		"""
		for target_num in num_list:
			target_string = str(target_num)
			for myname, param in self.features.named_parameters():
				if myname.startswith(target_string):
					param.requires_grad = is_training

	def train_features_partially_in(self, num_list, is_training):
		"""	Make the features trainable or not.

		Keyword arguments:
		num_list -- the list of layer numbers in features
		is_training -- the flag (train: True)
		"""
		for target_num in num_list:
			target_string = str(target_num)
			for myname, param in self.features.named_parameters():
				if target_string in myname:
					param.requires_grad = is_training
