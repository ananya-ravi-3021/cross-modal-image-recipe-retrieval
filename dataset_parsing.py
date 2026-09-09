import os
import pandas as pd
import numpy as np
import warnings

from pathlib import Path
from utils import RestrictionsInfo
from imageio import imread as imread

from utils import *

# Root of the downloaded dataset folders (food_10k, food_5k, food_13k, ...).
# Defaults to the current directory; override if your copy lives elsewhere:
#     export DATASET_ROOT=/path/to/datasets
dir = Path(os.environ.get('DATASET_ROOT', '.'))

# image directories for each source dataset
food_10k_images = str(dir/'food_10k/food_10k_images') + '/'
food_5k_images = str(dir/'food_5k/Recipes5k/images') + '/'
food_13k_images = str(dir/'food_13k/images/food_images') + '/'

# dictionarity of allergen/direct labels
restrictions_dict = RestrictionsInfo().getRestrictions()

# read dataset information
food_10k_df = pd.read_json(dir/'food_10k/food_10k_files/food_10k_info.json')
food_5k_df = pd.read_json(dir/'food_5k/Recipes5k_files/food_5k_info.json')
food_13k_df = pd.read_json(dir/'food_13k/food_13k_info.json')
minority_df = pd.read_json(dir/'food_1M/recipe_1M+/recipe_1M+_minority_info.json')

# mapping allergies to ingredients
warnings.filterwarnings('ignore')
def createLabels(data_frame, param):
    image_info = data_frame[["image_path", "title", "ingredients_list"]]
    restrictions = restrictions_dict.keys()
    image_info["image_path"] = param + data_frame["image_path"]

    for key in restrictions:
        image_info[key] = np.zeros(len(data_frame))

    count = 0
    for recipe in image_info["ingredients_list"]:
        for ingredient in recipe:
            for key in restrictions: 
                for item in restrictions_dict[key]:
                    if item in ingredient.lower():
                        image_info[key].iloc[count] = 1
        count += 1
        print(count)

    return image_info

# converting the program into csv files
def createCSVFiles(data_frame):
    train_size = 9 * len(data_frame) // 10

    train = data_frame.iloc[0: train_size]
    validate = data_frame.iloc[train_size:]

    data_frame.to_csv(all_supervised_path, index=False)
    train.to_csv(train_supervised_path, index=False)
    validate.to_csv(validate_supervised_path, index=False)

# initialisation process
df1 = createLabels(food_10k_df, food_10k_images)
df2 = createLabels(food_5k_df, food_5k_images)
df3 = createLabels(food_13k_df, food_13k_images)
df4 = createLabels(minority_df, "")
result_df = pd.concat([df1, df2, df3, df4]).sample(frac=1).reset_index(drop=True)
createCSVFiles(result_df)