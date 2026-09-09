import pandas as pd
import json
import os
from utils import * 
import csv

# read json file in
data_path = all_supervised_path

with open(data_path, 'r') as file:
    reader = csv.reader(file)
    # Skip header if present
    next(reader)  
    for line_number, row in enumerate(reader):
        if len(row) != 13:
            print(row)
            print(f"Skipping line {line_number + 1}: Expected 13 fields, saw {len(row)}")
            continue

def clone():
    # create dataframe for analysis
    data = pd.read_csv(data_path).iloc[19995:]

    # minority class labels
    minority_list = ["shellfish", "finned fish", "soy"]

    # clone minority recipes to csv
    for index, row in data.iterrows():
        print(index)
        
        is_minority = False
        
        for label in minority_list:
            if row[label] == 1.0:
                is_minority = True
                break
        
        # clone the minority recipe 3x
        if is_minority:
            row_df = row.to_frame().T
            for i in range(0, 2):
                row_df.to_csv(data_path, mode='a', header=False, index=False)
            
def split():        
    data = pd.read_csv(data_path)

    train_size = 9 * len(data) // 10

    train = data.iloc[0: train_size]
    validate = data.iloc[train_size:]

    train.to_csv(train_supervised_path, index=False)
    validate.to_csv(validate_supervised_path, index=False)

# clone()
split()
