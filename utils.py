import os
import os.path as osp

import pandas as pd
import numpy as np

class RestrictionsInfo:
    def __init__(self):
        # allergy/diet information
        # implementation using dictionary

        self.diet_dict = {
            "dairy": {"chocolate", "mascarpone", "butter", "cream", "milk", "cheese", "yogurt", "nutella"},
            "egg" : {"egg", "egg white", "egg yolk"},
            "gluten" : {"bread", "flour", "cereal", "pasta", "wheat noodle", "couscous", "semolina", "rye", "crust", "oat", "crackers"},
            "sugar" : {"sugar"},
            "peanut" : {"peanut"},
            "tree nut": {"nut", "almond", "cashew", "macadamia", "pistachio", "brazil nut", "walnut", "pine nut"},
            "shellfish" : {"shrimp", "prawn", "crab", "lobster", "clams", "mussels", "oysters", "scallops", "octopus", "squid", "abalone", "snail"},
            "finned fish" : {"fish", "tuna", "salmon", "halibut", "catfish", "cod"},
            "soy" : {"tofu", "soy milk", "edamame", "miso", "tempeh", "soy sauce"},
            "meats": {"chicken", "beef", "pork", "gelatin", "mutton", "lamb", "duck", "goose", "goat", "turkey", "ham", "steak"}
        }

        '''
        self.diet_dict = {
            "dairy": {"chocolate", "mascarpone", "butter", "cream", "milk", "cheese", "yogurt", "nutella"},
        }
        '''

    # getter methods  
    def getRestrictions(self):
        return self.diet_dict

# base directories - override with environment variables if your copies live elsewhere:
#     export DATA_DIR=/path/to/data_files
#     export MODEL_DIR=/path/to/model_states
#     export PKL_DIR=/path/to/gcn_pkl
DATA_DIR = os.environ.get('DATA_DIR', 'data_files')
MODEL_DIR = os.environ.get('MODEL_DIR', 'model_states')
PKL_DIR = os.environ.get('PKL_DIR', osp.join('model', 'gcn_pkl'))

# relevant paths
all_supervised_path = osp.join(DATA_DIR, 'all_supervised_data')
all_cloned_path = osp.join(DATA_DIR, 'all_cloned_data')
train_supervised_path = osp.join(DATA_DIR, 'train_supervised_data')
validate_supervised_path = osp.join(DATA_DIR, 'validate_supervised_data')
train_ssl_path = osp.join(DATA_DIR, 'semi_supervised_data')
validate_path = osp.join(DATA_DIR, 'validate_supervised_data')
ssl_storage_path = osp.join(DATA_DIR, 'semi_supervised_data')
model_path = osp.join(MODEL_DIR, 'curr_model')
model_supervised_path = osp.join(MODEL_DIR, 'supervised_model')

prob_pkl = osp.join(PKL_DIR, 'cooccurrence_data.pkl')
feature_pkl = osp.join(PKL_DIR, 'feature_data.pkl')