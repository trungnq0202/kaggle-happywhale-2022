import csv
import sys
import torch
from cv2 import mean
from matplotlib import transforms
from numpy import std
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
import torchvision.transforms as tf
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

sys.path.append("./libs")

class HappyWhaleDataset(data.Dataset):
    def __init__(
        self, csv_path, data_dir ="../../data/happy-whale-and-dolphin", IMG_SIZE=256, is_train=True
    ):
        
        transform_train = A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=60,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), 
                    contrast_limit=(-0.1, 0.1),
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2()
            ], p=1.)

        transform_validation = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ], p=1.)

        self.data_dir = data_dir
        self.data = pd.read_csv(csv_path)
        self.IMG_SIZE = IMG_SIZE
        # self.file_names = self.data['file_path'].values
        # self.labels = self.data['individual_id'].values
        self.transforms = transform_train if is_train else transform_validation
        self.is_train = is_train
        
        # self.df = pd.read_csv(f"{self.data_dir}/train.csv")
        # self.df['file_path'] = self.df['image'].apply(lambda x: f"{self.data_dir}/{x}")
        # encoder = LabelEncoder()
        # self.df['individual_id'] = encoder.fit_transform(self.df['individual_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # img_path = self.file_names[index]
        path, label = self.data.image.values[index], self.data.individual_id.values[index]
        path = os.path.join(self.data_dir, path)
        # img = cv2.imread(img_path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)['image']

        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    train_dataset = HappyWhaleDataset(
        csv_path="../../lists/train.csv",
        data_dir="../../data/happy-whale-and-dolphin/train_images/",
        IMG_SIZE=256,
        is_train=True
    )

    # val_dataset = HappyWhaleDataset(
    #     csv_path="../../lists/val.csv",
    #     data_dir="../../data/happy-whale-and-dolphin/test_images/",
    #     IMG_SIZE=256,
    #     is_train=False
    # )

    print(train_dataset[0])
    # print(val_dataset[0])