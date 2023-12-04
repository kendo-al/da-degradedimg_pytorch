""" helper functions for dgeraded images."""
from datahandler import degcifar, degtinyimagenet
import numpy as np

def get_type_range(degtype):
	""" Get degradation type and range.

	Keyword arguments:
	degtype -- the type of degradation
	"""
	if degtype == "jpeg":
		deg_type = "jpeg"
		deg_range = [1, 101]
	elif degtype == "noise":
		deg_type = "noise"
		deg_range = [0, 50]
	elif degtype == "blur":
		deg_type = "blur"
		deg_range = [0, 50]
	else:
		deg_type = "saltpepper"
		deg_range = [0, 25]
	return deg_type, deg_range

def get_minmax_normalizedlevel(deg_type):
	""" Min and Max of normalized degradation levels.

	Keyword arguments:
	deg_type -- the type of degradation
	"""
	if deg_type == 'jpeg':
		ret_adj = 100.0
		max_l = 101.0
		min_l = 1.0
	elif deg_type == 'noise':
		ret_adj = 255.0
		max_l = 50.0
		min_l = 0.0
	elif deg_type == 'blur':
		ret_adj = 100.0
		max_l = 50.0
		min_l = 0.0
	elif deg_type == 'saltpepper':
		ret_adj = 100.0
		max_l = 5.0
		min_l = 0.0
	else:
		ret_adj = 1.0
		max_l = 1.0
		min_l = 0.0

	return min_l/ret_adj, max_l/ret_adj

def fix_seed_noise_sl(is_fixed):
	""" Fix the seed of Gaussian and Binomial distributions.
	This is only used for evalution purpose.
	If you fix the seed, please do not forget to unfix the seed.

	Keyword arguments:
	is_fixed -- the flag
	"""
	if is_fixed:
		np.random.seed(seed=301)
	else:
		np.random.seed(seed=None)

def get_datasetclass(data_name = "CIFAR10", train = True, transform = None, download = True, deg_transform = None):
	""" Get a dataset object.

	Keyword arguments:
	data_name -- the dataset name
	train -- the flag to use training data
	transform -- the data augmentations for clean data
	degtransform -- the data augmentation for degraded data including degradation
	"""
	if data_name == "CIFAR10":
		datasetclass = degcifar.DegCIFAR10( root = '../datasets/', train = train,  transform = transform, download = download, is_to_tensor = True, is_target_to_tensor = True, is_normalized = True, deg_transform = deg_transform)
	elif data_name == "CIFAR100":
		datasetclass = degcifar.DegCIFAR100( root = '../datasets/', train = train,  transform = transform, download = download, is_to_tensor = True, is_target_to_tensor = True, is_normalized = True, deg_transform = deg_transform)
	elif data_name == "TINY":
		datasetclass = degtinyimagenet.DegTINYIMAGENET( root = '../datasets/', train = train, transform = transform, download = download, is_to_tensor = True, is_target_to_tensor = True, is_normalized = True, deg_transform = deg_transform, validation = not(train))
	else:
		datasetclass = None
	return datasetclass

def get_type_list(degtype):
	""" Get degradation type and range.

	Keyword arguments:
	degtype -- the type of degradation
	"""
	if degtype == "jpeg":
		deg_type = "jpeg"
		deg_list = [10, 30, 50, 70, 90, 101]
	elif degtype == "noise":
		deg_type = "noise"
		deg_list = [0.0, np.sqrt(0.05)*255, np.sqrt(0.1)*255, np.sqrt(0.15)*255, np.sqrt(0.2)*255, np.sqrt(0.25)*255]
	elif degtype == "blur":
		deg_type = "blur"
		deg_list = [0.0, 10., 20., 30., 40., 50.]
	else:
		deg_type = "saltpepper"
		deg_list = [0.0, 5., 10., 15., 20., 25.]

	return deg_type, deg_list

