import os
import os.path as osp
import pandas as pd
import numpy as np
import warnings

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models as models

from food_dataset import FoodDataset

from loss.asl_loss import AsymmetricLossOptimized
from loss.caal import CAAL

from model.naive_model import CNN_Class
from model.resnet_csra import ResNet_CSRA
from model.gcn import *

from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

from torch.cuda.amp import autocast, GradScaler

from utils import *

import time

classes_number = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory holding the dataset csv files. Download the dataset separately (see
# README) and either place it in ./data_files or point DATA_DIR at your copy:
#     export DATA_DIR=/path/to/data_files
DATA_DIR = os.environ.get('DATA_DIR', 'data_files')

# reading allergy data files in
train_supervised = pd.read_csv(osp.join(DATA_DIR, 'train_supervised_data'))
# train_semi_supervised = pd.read_csv(osp.join(DATA_DIR, 'semi_supervised_data'))
validate = pd.read_csv(osp.join(DATA_DIR, 'validate_supervised_data'))

# creating model
model = CNN_Class().to(device)
# model = ResNet_CSRA(num_heads=1, lam=0.2, num_classes=classes_number, cutmix=None).to(device)
# thold = 0.0375
# model = gcn_resnet101(thold, num_classes=10, allergy_data=train_supervised).to(device)

# identifying hyperparameter values
batch_size = 16
lr = 0.0001
max_lr = 0.001
weight_decay = 0.01
epochs = 5

# creating the training and test dataset
train_dataset = FoodDataset("train", train_supervised)
validate_dataset = FoodDataset("val", validate)

# creating the training, validation and test data loaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False, num_workers=4)
validate_loader = DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers=4)

# loss function
# loss_function = nn.BCELoss().to(device)
# loss_function = AsymmetricLossOptimized(gamma_neg=5, gamma_pos=2, clip=0.1, disable_torch_grad_focal_loss=True).to(device)
loss_function = CAAL(10, 6.0, 2.0, 12.0, 0.8, 0.2)

# setting optimiser and scheduler
optimiser = torch.optim.SGD(model.parameters(), lr, weight_decay)
scheduler = OneCycleLR(optimiser, max_lr, epochs * len(train_loader))

def train_implementation():
    highest_mAP = 0
    highest_macro_f1 = 0
    highest_micro_f1 = 0

    for epoch in range(epochs):
        
        train_loss = 0.0
        model.train()

        for index, data in enumerate(train_loader):
            start_time = time.time()

            images, labels = data['image'].to(device), data['image_label'].to(device)
            
            optimiser.zero_grad()

            # forward pass and calculate loss function
            # outputs, loss = model(images, labels)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # calculating train loss
            train_loss += loss.item()

            # backward and optimise
            loss.backward()
            optimiser.step()
            scheduler.step()

            end_time = time.time()

        # calculate train loss
        train_loss = train_loss/len(train_loader)
        print("train loss" , train_loss)
            
        model.eval()
        mAP, macro_f1, micro_f1 = validate_implementation(model)

        if mAP > highest_mAP:
            highest_mAP = mAP
            torch.save(model.state_dict(), model_supervised_path)

        if macro_f1 > highest_macro_f1:
            highest_macro_f1 = macro_f1

        if micro_f1 > highest_micro_f1:
            highest_micro_f1 = micro_f1
        
    print("Highest mAP", highest_mAP * 100, "%")
    print("Highest Macro F1", highest_macro_f1 * 100, "%")
    print("Highest Micro F1", highest_micro_f1 * 100, "%")
    
warnings.filterwarnings('ignore')
def validate_implementation(model):
    # validation check
    ap_scores_total = list()
    f1_scores_macro_total = list()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
 
    with torch.no_grad():
    # validation method - calulcate ap score and f1 score
        for index, data in enumerate(validate_loader):
            
            images, labels = data['image'].to(device), data['image_label'].to(device)

            # outputs, loss = model(images, labels)

            outputs = model(images)
            
            ap_scores_batch = list()
            f1_macro_scores_batch = list()

            for i in range(classes_number):
                class_pred = outputs[:, i].cpu().numpy()
                class_actual = labels[:, i].cpu().numpy()

                # ap score per class
                ap_score = average_precision_score(class_actual, class_pred)
                ap_scores_batch.append(ap_score)

                # macro f1 score per class
                f1_macro = f1_score(class_actual, (class_pred >= 0.5).astype(int))
                f1_macro_scores_batch.append(f1_macro)

                # Update true positives, false positives, and false negatives for micro F1 calculation
                true_positives += np.sum((class_pred >= 0.5) & (class_actual == 1))
                false_positives += np.sum((class_pred >= 0.5) & (class_actual == 0))
                false_negatives += np.sum((class_pred < 0.5) & (class_actual == 1))

            # ap score for that batch
            ap_scores_total.extend(ap_scores_batch)

            # f1 macro score for that batch
            f1_scores_macro_total.extend(f1_macro_scores_batch)

    # calculate mAP score at the end
    mAP_score = np.mean(ap_scores_total)

    # Calculate overall macro F1 score
    overall_macro_f1_score = np.mean(f1_scores_macro_total)

    # Calculate overall macro F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    overall_micro_f1_score = 2 * (precision * recall) / (precision + recall)

    print(" val mAP accuracy ", mAP_score, " val macro F1 accuracy ", overall_macro_f1_score, 
          " val micro F1 accuracy ", overall_micro_f1_score)

    return (mAP_score, overall_macro_f1_score, overall_micro_f1_score)

train_implementation()