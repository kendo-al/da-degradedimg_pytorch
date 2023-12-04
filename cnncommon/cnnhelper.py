# Helper functions for trainings and evalutions
import os
import torch
import torch.nn as nn
import csv
import numpy as np
from datetime import datetime

def save_model(root_dir = None, model_name = 'cnnmodel', epoch = 0, target_dict = None):
	""" Save a model checkpoint as *.pth.
	
	Keyword arguments:
	root_dir -- the name of a root directory
	model_name -- the file name ({model_name}_{epoch}.pth, default cnnmodel)
	epoch -- the epoch number (default 0)
	target_dict -- the model checkpoint dictionary
	"""
	file_name = '{}_{}.pth'.format(model_name, epoch)
	path = os.path.join(root_dir, file_name)
	os.makedirs(root_dir, exist_ok=True)
	torch.save(target_dict, path)

def load_model(root_dir = None, model_name = 'cnnmodel', epoch = 0):
	"""	Load a model checkpoint.

	Keyword arguments:
	root_dir -- the name of a root directory
	model_name -- the file name ({model_name}_{epoch}.pth, default cnnmodel)
	epoch -- the epoch number (default 0)
	"""
	file_name = '{}_{}.pth'.format(model_name, epoch)
	path = os.path.join(root_dir, file_name)
	if os.path.exists(path):
		checkpoint = torch.load(path)
	else:
		checkpoint = None
	return checkpoint

def save_csv(root_dir = None, file_name = None, target_list = None, header_list = None):
	"""	Save the training progress.

	Keyword arguments:
	root_dir -- the name of a root directory
	file_name -- the file name ({file_name}.csv)
	target_list -- the output list
	header_list -- the header list
	"""
	path = os.path.join(root_dir, file_name)
	if not(os.path.exists(path)):
		os.makedirs(root_dir, exist_ok=True)
		with open(path, 'a') as csvfile:
			writer = csv.writer(csvfile, lineterminator='\n')
			if header_list is not None:
				writer.writerow(header_list)
			writer.writerow(target_list)
	else:
		with open(path, 'a') as csvfile:
			writer = csv.writer(csvfile, lineterminator='\n')
			writer.writerow(target_list)

def get_logmodel_dir(rootname = 'models', deg_type = "jpeg", model_name = None, exp_string = "CLEAN"):
	""" Get the directory of saved models.

	Keyword arguments:
   	rootname -- the root directory's name
	deg_type -- degradation types: jpeg, noise, blur, saltpepper (default jpeg)
	model_name -- model types: VGG16, ResNet50, ResNet2_56, Shake110-270 
	exp_string -- experiments: "CLEAN", "MIXED", "CUTBLUR" (default "CLEAN")
	"""
	if exp_string == "CLEAN":
		exp_dir = "clean"
	elif exp_string == "MIXED":
		exp_dir = "mixed"
	elif exp_string == "CUTBLUR":
		exp_dir = "cutblur"
	else:
		msg = 'Unsupported degradation flag.'
		raise TypeError(msg)

	dir = os.path.join(rootname, deg_type, model_name, exp_dir)

	return dir

def num_correctness(outputs, labels):
	""" Return the number of correct predictions.
	
	Keyword arguments:
	outputs -- predictions
	labels -- correct labels
	"""
	return (outputs.max(1)[1] == labels).sum().item()

def timestamp():
	"""	Timestamp."""
	return datetime.now().strftime("%Y/%m/%d %H:%M:%S")

def get_eval_dir_filename(rootname = 'evaluations', deg_type = "jpeg", model_name = None, exp_string = "CLEAN", epoch = None):
	""" Get the directory and filename for evaluations.

	Keyword arguments:
	rootname -- the root directory's name (default evaluations)
	deg_type -- degradation types: jpeg, noise, blur, saltpepper (default jpeg)
	model_name -- model types: VGG16, ResNet50, ResNet2_56, Shake110-270
	exp_string -- experiments: "CLEAN", "MIXED", "CUTBLUR" (default "CLEAN")
	epoch -- the epoch number
	"""
	if exp_string == "CLEAN":
		exp_dir = "clean"
	elif exp_string == "MIXED":
		exp_dir = "mixed"
	elif exp_string == "CUTBLUR":
		exp_dir = "cutblur"
	else:
		msg = 'Unsupported experiment!'
		raise TypeError(msg)

	file_name = "res_" + exp_dir

	if epoch is not None:
		file_name += "_"
		file_name += str(epoch)
	file_name += ".csv"

	dir = os.path.join(rootname, deg_type, model_name)

	return dir, file_name
