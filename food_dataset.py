import cv2
import torch
import torchvision.transforms as transforms
import warnings
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, dataset, csv):
        self.csv = csv

        self.images = self.csv['image_path']
        self.labels = self.csv.drop(['image_path', 'title', 'ingredients_list'], axis=1)
        
        # for both labelled training and semi supervised data
        if dataset == "train" or dataset == "unlabelled":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
            ])

        if dataset == "val":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.csv)
    
    # generate image given the specified path
    def __getitem__(self, index):
        image = self.images.iloc[index]
        image = cv2.imread(image)
        image = self.transform(image)
        image_labels = self.labels.iloc[index].to_numpy()[:]

        warnings.filterwarnings("ignore")
        image = torch.tensor(image, dtype=torch.float32)
        image_labels = torch.tensor(image_labels, dtype=torch.float32)
        sample_dict = {"image": image, "image_label": image_labels}
        return sample_dict


        
        

        

     
