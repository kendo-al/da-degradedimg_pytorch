"""
Evaluate CNNs
"""
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import json
from numpy.random import RandomState

from cnnmodel import cnn_builder
from datahandler import degimagedata, degtransforms
from cnncommon.cnnhelper import load_model, save_csv, get_logmodel_dir, num_correctness, get_eval_dir_filename

# Sys args
last_epoch = 500 # Last epoch, should be "0" in the case of initial training.
exp_string = "CUTBLUR" # Select a key string from "CLEAN", "MIXED", and "CUTBLUR"
exp_no = 0 # 0: VGG16 + CIFAR10, 1: ResNet50 + CIFAR10, 2: ShakePyramidNet + CIFAR10, 3: ShakePyramidNet + CIFAR100, 4: ResNet56 + TINY
deg_kind = "jpeg" # jpeg, noise, blur, saltpepper

# Meta settings
batch_size = 100
num_workers = 1
headerlist=["Deg_level","CEL","ACC"]
is_cifar = False

# Configuration for experiments
open_config = None
if deg_kind == "noise":
	open_config = open('experiments/cl_exp_config_noise.json','r')
elif deg_kind == "blur":
	open_config = open('experiments/cl_exp_config_blur.json','r')
elif deg_kind == "saltpepper":
	open_config = open('experiments/cl_exp_config_saltpepper.json','r')
else: #JPEG
	open_config = open('experiments/cl_exp_config.json','r')

try:
	model_config = json.load(open_config)
finally:
	open_config.close()

d = model_config[exp_string][exp_no]

# Degradation method for evaluations 
test_deg_trans_org = []
if(d["data_name"] == "CIFAR10" or d["data_name"] == "CIFAR100"):
    is_cifar = True
    deg_type, deg_range = degimagedata.get_type_range(d["deg_type"])
    deg_levs = range(deg_range[0],deg_range[1]+1) 	
elif(d["data_name"] == "TINY"):
    is_cifar = True
    deg_type, deg_range = degimagedata.get_type_range(d["deg_type"])
    deg_levs = range(deg_range[0],deg_range[1]+1)
else:
    msg = 'Please add new data augmenations. Currently, unspoorted dataset.'
    raise TypeError(msg)

# Get the root of outputs
dir, filename = get_eval_dir_filename(rootname = 'evaluations' + d["data_name"], deg_type = d["deg_type"], model_name = d["model_name"], exp_string = exp_string, epoch = last_epoch)

# load a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_dir = get_logmodel_dir(rootname = 'models_' + d["data_name"] , deg_type = d["deg_type"], model_name = d["model_name"], exp_string = exp_string)
last_checkpoint = load_model(root_dir = root_dir, epoch = last_epoch)
deg_nw = cnn_builder.buildNetwork(model_name = d["model_name"], num_of_classes = d["num_classes"], num_of_features = d["num_features"], init_weights = True, is_cifar = is_cifar)
deg_nw.load_state_dict(last_checkpoint["model_state_dict"])
net = deg_nw.to(device)
net.eval()

# Loss function
cross_entoropy_loss = nn.CrossEntropyLoss()

# Fix the seeds of Guassian and binomial distributions for degradation
#degimagedata.fix_seed_noise_sl(is_fixed = True) # classical numpy rundom generation until ver. 1.16
rs = RandomState(seed = 301)

print("----- Evaluation START -----")
print(*headerlist)

for lev in deg_levs:
    # Data generation
    test_deg_trans = test_deg_trans_org + [degtransforms.DegradationApply(deg_type = deg_type, deg_range = [lev, lev], deg_list = None, rs = rs)]
    test_deg_transform = transforms.Compose(test_deg_trans)
    test_dataset = degimagedata.get_datasetclass(data_name = d["data_name"], train = False, transform = None, deg_transform = test_deg_transform, download = False)
    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    # Initialize variabels
    val_loss = 0
    val_cross_entoropy_loss = 0
    val_acc = 0
 
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, leave=False)):
            images=images.to(device)
            labels=labels.to(device)
            val_num_of_samples = labels.size(0)

            # Predictions
            pred_out = net(images)
            val_acc += num_correctness(pred_out, labels)

            # Calculate losses
            ce = cross_entoropy_loss(pred_out, labels)
            val_cross_entoropy_loss += ce.item()*val_num_of_samples

		# Results for each degradation level
        val_cross_entoropy_loss /= len(test_loader.dataset)
        val_acc /= len(test_loader.dataset)

    resultlist=[lev,val_cross_entoropy_loss,val_acc]
    save_csv(root_dir = dir, file_name = filename, target_list = resultlist, header_list = headerlist)
    print(*resultlist)

# Unfix the seeds of Guassian and binamial distributions for degradation
#degimagedata.fix_seed_noise_sl(is_fixed = False) # classical numpy rundom generation until ver. 1.16
print("----- Evaluation END -----")
