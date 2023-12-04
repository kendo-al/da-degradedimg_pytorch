"""
Train CNNs
"""
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torchsummary import summary # https://github.com/sksq96/pytorch-summary
import json
import torch.nn.functional as F
import numpy as np

from cnnmodel import cnn_builder
from datahandler import degimagedata, degaugtransforms, degtransforms, cutouttransforms
from cnncommon.cnnhelper import save_model, load_model, save_csv, get_logmodel_dir, num_correctness, timestamp
from radam import RAdam


# Sys args
num_epochs = 500 # Number of epochs
last_epoch = 0 # Last epoch, should be "0" in the case of initial training.
exp_string = "CUTBLUR" # Select a key string from "CLEAN", "MIXED", and "CUTBLUR"
exp_no = 0 # 0: VGG16 + CIFAR10, 1: ResNet50 + CIFAR10, 2: ShakePyramidNet + CIFAR10, 3: ShakePyramidNet + CIFAR100, 4: ResNet56 + TINY
view_model = "train" # train, view
deg_kind = "jpeg" # jpeg, noise, blur, saltpepper

# Meta settings
batch_size = 128 #CIFAR:128, TINY:128
test_batch_size = 100 #CIFAR:100, TINY:100
num_workers = 2
headerlist=["Epoch","Time","Train_loss","Train_CEL","Train_ACC","Val_loss","Val_CEL","Val_ACC"]
optimizer_reset = False
is_lrSchedule = False
is_cifar = False

# Configuration for experiments
open_config = None
deg_range1 = None
if deg_kind == "noise":
	open_config = open('experiments/cl_exp_config_noise.json','r')
	deg_range1=[1,50]
elif deg_kind == "blur":
	open_config = open('experiments/cl_exp_config_blur.json','r')
	deg_range1=[1,50]
elif deg_kind == "saltpepper":
	open_config = open('experiments/cl_exp_config_saltpepper.json','r')
	deg_range1=[1,25]
else: #JPEG
	open_config = open('experiments/cl_exp_config.json','r')
	deg_range1=[1,100]

try:
	model_config = json.load(open_config)
finally:
	open_config.close()

d = model_config[exp_string][exp_no]

# Data generation
# (1) Geometric tansformation for raw data.
transform = transforms.Compose([transforms.RandomHorizontalFlip()])
# (2) Degradation and some transforms after degradation
deg_trans = None
test_deg_trans = None
if(d["data_name"] == "CIFAR10" or d["data_name"] == "CIFAR100"):
	is_cifar = True
	batch_size = 128
	deg_type, deg_range = degimagedata.get_type_range(d["deg_type"])
	test_deg_trans = [degtransforms.DegradationApply(deg_type = deg_type, deg_range = deg_range, deg_list = None)] 	
	if exp_string == "MIXED":
		# Mixed-Training
		#deg_trans = [degtransforms.DegradationApply(deg_type = deg_type, deg_range = deg_range, deg_list = None), transforms.RandomCrop(size=32, padding=4, fill=0, padding_mode='constant')]
		# Mixed-Training with Random Erasing
		deg_trans = [degtransforms.DegradationApply(deg_type = deg_type, deg_range = deg_range, deg_list = None), cutouttransforms.CutOutApply(cut_prob=2.0, cutratio_min=0.125, cutratio_max=0.75), transforms.RandomCrop(size=32, padding=4, fill=0, padding_mode='constant')]
	elif exp_string == "CUTBLUR":
		# CutBlur
		#deg_trans = [degaugtransforms.CutDegradationApply(deg_type = deg_type,deg_ranges=[deg_range1],deg_lists=None,cut_ratios=[0.125,0.75],is_clean_bkg=False, is_clean_cutout=False), transforms.RandomCrop(size=32, padding=4, fill=0, padding_mode='constant')]
		# Proposed
		deg_trans = [degaugtransforms.CutDegradationApply(deg_type = deg_type,deg_ranges=[deg_range1],deg_lists=None,cut_ratios=[0.125,0.75],is_clean_bkg=False, is_clean_cutout=True), transforms.RandomCrop(size=32, padding=4, fill=0, padding_mode='constant')]
	else: # Clean
		deg_trans = [transforms.RandomCrop(size=32, padding=4, fill=0, padding_mode='constant')]
		test_deg_trans = None 
elif(d["data_name"] == "TINY"):
	is_cifar = True # True for TINY ImageNet
	batch_size = 128
	deg_type, deg_range = degimagedata.get_type_range(d["deg_type"])
	test_deg_trans = [degtransforms.DegradationApply(deg_type = deg_type, deg_range = deg_range, deg_list = None)] 	
	if exp_string == "MIXED":
		# Mixed-Training
		#deg_trans = [degtransforms.DegradationApply(deg_type = deg_type, deg_range = deg_range, deg_list = None), transforms.RandomCrop(size=64, padding=4, fill=0, padding_mode='constant')]
		# Mixed-Training with Random Erasing
		deg_trans = [degtransforms.DegradationApply(deg_type = deg_type, deg_range = deg_range, deg_list = None), cutouttransforms.CutOutApply(cut_prob=2.0, cutratio_min=0.125, cutratio_max=0.75), transforms.RandomCrop(size=64, padding=4, fill=0, padding_mode='constant')]
	elif exp_string == "CUTBLUR":
		# CutBlur
		#deg_trans = [degaugtransforms.CutDegradationApply(deg_type = deg_type,deg_ranges=[deg_range1],deg_lists=None,cut_ratios=[0.125,0.75],is_clean_bkg=False, is_clean_cutout=False), transforms.RandomCrop(size=64, padding=4, fill=0, padding_mode='constant')]
		# Proposed
		deg_trans = [degaugtransforms.CutDegradationApply(deg_type = deg_type,deg_ranges=[deg_range1],deg_lists=None,cut_ratios=[0.125,0.75],is_clean_bkg=False, is_clean_cutout=True), transforms.RandomCrop(size=64, padding=4, fill=0, padding_mode='constant')]
	else: # Clean
		deg_trans = [transforms.RandomCrop(size=64, padding=4, fill=0, padding_mode='constant')]
		test_deg_trans = None
else:
	msg = 'Please add new data augmenations. Currently, unspoorted dataset.'
	raise TypeError(msg)

if deg_trans is None:
	deg_transform = None
else:
	deg_transform = transforms.Compose(deg_trans)

if test_deg_trans is None:
	test_deg_transform = None
else:
	test_deg_transform = transforms.Compose(test_deg_trans)

train_dataset = degimagedata.get_datasetclass(data_name = d["data_name"], train = True, transform = transform, deg_transform = deg_transform)
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
test_dataset = degimagedata.get_datasetclass(data_name = d["data_name"], train = False, transform = None, deg_transform = test_deg_transform)
test_loader = torch.utils.data.DataLoader( test_dataset, batch_size = test_batch_size, shuffle = False, num_workers = num_workers)

# Load a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_dir = get_logmodel_dir(rootname = 'models_' + d["data_name"] , deg_type = d["deg_type"], model_name = d["model_name"], exp_string = exp_string)
last_checkpoint = load_model(root_dir = root_dir, epoch = last_epoch)

# Model initilization
deg_nw = cnn_builder.buildNetwork(model_name = d["model_name"], num_of_classes = d["num_classes"], num_of_features = d["num_features"], init_weights = True, is_cifar = is_cifar)
if last_checkpoint is not None: # Continue the training.
	deg_nw.load_state_dict(last_checkpoint["model_state_dict"])

# Send the model to a device
net = deg_nw.to(device)

# Optimizer setting
if d["model_name"] == "VGG16":
	optimizer = RAdam(net.parameters(), lr = 1e-3, weight_decay = 1e-4)
elif d["model_name"] == "ResNet50" or d["model_name"] == "ResNet2_56":
	if d["data_name"] == "TINY":
		is_lrSchedule = True
		optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0005, nesterov = True)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120, 160, 200, 240, 280], gamma = 0.2)
	else:
		optimizer = RAdam(net.parameters(), lr = 1e-3, weight_decay = 1e-4)
elif d["model_name"] == "Shake110-270":
	if d["data_name"] == "CIFAR10":
		is_lrSchedule = True
		optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [75, 150], gamma = 0.1)
	elif d["data_name"] == "CIFAR100":
		is_lrSchedule = True
		optimizer = torch.optim.SGD(net.parameters(), lr = 0.5, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 225], gamma = 0.1)
	else:
		optimizer = RAdam(net.parameters(), lr = 1e-3, weight_decay = 1e-4)	

if ((last_checkpoint is not None) and (not(optimizer_reset))):
	optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])

# Loss function
cross_entoropy_loss = nn.CrossEntropyLoss()

if view_model == "view":
	print('-----Model Summary-----')
	input_size = train_dataset.getCHW()
	summary(net, input_size)
	sys.exit()

print('-----Training start at {}-----'.format(timestamp()))
print(*headerlist)
for epoch in range(num_epochs):
	train_loss = 0
	train_cross_entoropy_loss = 0
	train_acc = 0
	val_loss = 0
	val_cross_entoropy_loss = 0
	val_acc = 0

	# ===== Start training =====
	net.train()

	for i, (images, labels) in enumerate(tqdm(train_loader, leave=False)):

		images = images.to(device)
		labels = labels.to(device)
		num_of_samples = labels.size(0)

		# Forward induction
		optimizer.zero_grad()
		pred_out = net(images)
		train_acc += num_correctness(pred_out, labels)

		# Backward induction
		loss = cross_entoropy_loss(pred_out, labels)
		train_cross_entoropy_loss += loss.item()*num_of_samples
		train_loss += loss.item()*num_of_samples
		loss.backward()
		optimizer.step()

	# Results for each epoch
	train_loss /= len(train_loader.dataset)
	train_cross_entoropy_loss /= len(train_loader.dataset)
	train_acc /= len(train_loader.dataset)

	# ===== Start testing =====
	net.eval()
	with torch.no_grad():
		for i, (images, labels) in enumerate(test_loader):
			images = images.to(device)
			labels = labels.to(device)
			val_num_of_samples = labels.size(0)

			# Predictions
			pred_out = net(images)
			val_acc += num_correctness(pred_out, labels)

			# Calculate losses
			ce = cross_entoropy_loss(pred_out, labels)
			val_cross_entoropy_loss += ce.item()*val_num_of_samples
			loss = ce
			val_loss += loss.item()*val_num_of_samples

		# Results for each epoch
		val_loss /= len(test_loader.dataset)
		val_cross_entoropy_loss /= len(test_loader.dataset)
		val_acc /= len(test_loader.dataset)

	loglist=[
		epoch+1+last_epoch,
		timestamp(),
		train_loss,
		train_cross_entoropy_loss,
		train_acc,
		val_loss,
		val_cross_entoropy_loss,
		val_acc
		]

	save_csv(root_dir = root_dir, file_name = "train_log.csv", target_list = loglist, header_list = headerlist)
	print(*loglist)

	if is_lrSchedule:
		scheduler.step()

save_model(root_dir = root_dir, model_name = 'cnnmodel', epoch = num_epochs + last_epoch, target_dict = {"model_state_dict":net.state_dict(),"optimizer_state_dict":optimizer.state_dict()})
