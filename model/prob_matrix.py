import numpy as np
import pandas as pd
import os
import pickle

from utils import *

train_supervised = pd.read_csv(train_supervised_path)

def create_cooccurence_matrix(allergy_data):
    # Initialize an empty cooccurrence matrix
    labels = allergy_data.columns[3:]
    co_matrix = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)

    # Populate the cooccurence matrix based on the DataFrame
    for i, row in allergy_data.iterrows():
        for label1 in labels:
            for label2 in labels:
                if label1 == label2:
                    continue
                if row[label1] == 1 and row[label2] == 1:
                    co_matrix.loc[label1, label2] += 1

    co_matrix = co_matrix.values
    label_counts = allergy_data.iloc[:, 3:].sum().reset_index(drop=True).values.astype(np.float64)

    return co_matrix, label_counts

# Assuming you have your allergy_data DataFrame ready
# Call the function to get the cooccurrence matrix and label counts
co_matrix, label_counts = create_cooccurence_matrix(train_supervised)

# Serialize the return information into a .pkl file
with open(PKL_DIR + 'cooccurrence_data.pkl', 'wb') as file:
    pickle.dump((co_matrix, label_counts), file)