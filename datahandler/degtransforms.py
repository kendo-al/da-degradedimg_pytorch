import torch
import numpy as np
import random
from PIL import Image, ImageStat, ImageChops
from datahandler import imagedegrade_im2 as degrade

def jpegcompresswithclean(img, jpeg_quality, rs = None):
	"""	Apply JPEG distortion to clean images.
	If JPEG quality factor is in [1, 100], JPEG distortion is applied.
	If it is not in [1, 100], clean image will be returned.
	Keyword arguments:
	img -- the clean PIL image
	jpeg_quality -- the JPEG quality factor
	"""
	if (jpeg_quality >= 1) and (jpeg_quality <= 100):
		ret_img = degrade.jpeg(img, jpeg_quality, rs)
	else:
		ret_img = img

	return ret_img

def degradation_function(deg_type):
	"""	Get a degradation function from imagedegrade_im2.
	
	Keyword arguments:
	deg_type -- the type of degradtion
	"""
	if deg_type == 'jpeg':
		ret_func = jpegcompresswithclean
		ret_adj = 1.0
	elif deg_type == 'noise':
		ret_func = degrade.noise
		ret_adj = 1.0
	elif deg_type == 'blur':
		ret_func = degrade.blur
		ret_adj = 10.0
	elif deg_type == 'saltpepper':
		ret_func = degrade.saltpepper
		ret_adj = 100.0
	else:
		msg = 'This degradation is not supported.'
		raise LookupError(msg)

	return ret_func, ret_adj

def mormalize_level(deg_type, level):
	"""	Normaliza degradation levels.

	Keyword arguments:
	deg_type -- the type of degradation
	level -- the degradation level
	"""
	if deg_type == 'jpeg':
		ret_adj = 100.0
	elif deg_type == 'noise':
		ret_adj = 255.0
	elif deg_type == 'blur':
		ret_adj = 10.0
	elif deg_type == 'saltpepper':
		ret_adj = 1.0
	else:
		ret_adj = 1.0
	ret = np.array([float(level)/ret_adj])
	ret.astype(np.float32)
	return ret

class DegradationApplyWithLevel(torch.nn.Module):
	""" Data augmentation of degradations.
	This transform returns not only a degraded image but also a degradation level.
	"""
	def __init__(self, deg_type, deg_range, deg_list, rs = None):
		""" Constructor.

		Keyword arguments:
		deg_type -- the type of degradtion
		deg_range -- the range of degradation levels
		deg_list -- the list of degradation levels
		rs -- Numpy randomstate
		"""
		super().__init__()
		self.deg_type = deg_type
		if deg_range is None and deg_list is None:
			msg = 'Both deg_range and deg_list do not have values.'
			raise TypeError(msg)
		elif (deg_range is not None) and (deg_list is not None):
			msg = 'deg_range or deg_list have values.'
			raise TypeError(msg)
		else:
			self.deg_range = deg_range
			self.deg_list = deg_list
		self.rs = rs
		self.deg_func, self.deg_adj = degradation_function(deg_type)

	def forward(self, img):
		""" Get a degraded image and a degradation level.

		Keyword arguments:
		img -- the clean PIL image
		"""
		if self.deg_range is not None:
			deg_lev = random.randint(self.deg_range[0], self.deg_range[1])
			if self.deg_adj > 1.0:
				deg_lev = deg_lev / self.deg_adj
		else:
			deg_lev = random.choice(self.deg_list)
			if self.deg_adj > 1.0:
				deg_lev = deg_lev / self.deg_adj

		return self.deg_func(img, deg_lev, self.rs), deg_lev

	def __repr__(self):
		return self.__class__.__name__ + '(deg_type={}, deg_range=({},{}))'.format(self.deg_type, self.deg_range[0], self.deg_range[1])

class DegradationApply(DegradationApplyWithLevel):
	""" Data augmentation of degradations."""
	def forward(self, img):
		""" Get a degraded image.

		Keyword arguments:
		img -- the clean PIL image
		"""
		deg_img, deg_lev = super().forward(img)
		return deg_img
