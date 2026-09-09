import pandas as pd
import numpy as np
import torch
import json
import os

from torchvision import models as models
from food_dataset import FoodDataset
from utils import *

from model.naive_model import CNN_Class
from model.resnet_csra import ResNet_CSRA
from model.gcn import *

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pseudo_label_data(begin, end):

    # model architecture
    model = CNN_Class().to(device)
    # model = ResNet_CSRA(num_heads=8, lam=0.1, num_classes=classes_number, cutmix=None)
    # model = gcn_resnet101(num_classes=10, t=0.4, allergy_data=train_supervised).to(device)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(model_path))

    # getting dataframe of unlabelled data
    unlabelled_data = pd.read_json('/home/ananya/Ananya_Ravi_FYP/food_1M/recipe_1M/recipe_1M_info.json').iloc[begin:end].reset_index(drop=True)

    # generating images and dataloader
    unlabelled_dataset = FoodDataset("unlabelled", unlabelled_data)
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size = 16, shuffle = False, num_workers=6)

    # labelling based on model prediction
    outputs = []

    for index, data in enumerate(unlabelled_loader):
        images = data['image'].to(device)
        with torch.no_grad():
            batch_outputs = model(images).cpu().detach().numpy()
        binary_outputs = np.where(batch_outputs >= 0.5, 1.0, 0.0)
        outputs.extend(binary_outputs.tolist())

    restrictions = RestrictionsInfo().getRestrictions().keys()
    pseudo_labels = pd.DataFrame(outputs, columns=restrictions)

    # combining new and existing pseudo-labelled data
    new_pseudo_data = pd.concat([unlabelled_data, pseudo_labels], axis=1)
    existing_pseudo_data = pd.read_csv(ssl_storage_path)
    
    pseudo_data = pd.concat([existing_pseudo_data, new_pseudo_data])
    pseudo_data.set_index('image_path', inplace=True)
    pseudo_data.to_csv(ssl_storage_path)
