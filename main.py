from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def make_data_tag():
    json_datatag = {
        0 : ['chevrolet', 'malibu', 'sedan', '2012_2016'],
        1 : ['chevrolet', 'malibu', 'sedan', '2017_2019'],
        2 : ['chevrolet', 'spark', 'hatchback', '2016_2021'],
        3 : ['chevrolet', 'trailblazer', 'suv', '2021_'],
        4 : ['chevrolet', 'trax', 'suv', '2017_2019'],
        5 : ['genesis', 'g80', 'sedan', '2016_2020'],
        6 : ['genesis', 'g80', 'sedan', '2021_'],
        7 : ['genesis', 'g80', 'suv', '2020_'],
        8 : ['hyundai', 'avante', 'sedan', '2011_2015'],
        9 : ['hyundai', 'avante', 'sedan', '2020_'],
        10 : ['hyundai', 'grandeur', 'sedan', '2011_2016'],
        11 : ['hyundai', 'grandstarex', 'van', '2018_2020'],
        12 : ['hyundai', 'ioniq', 'hatchback', '2016_2019'],
        13 : ['hyundai', 'sonata', 'sedan', '2004_2009'],
        14 : ['hyundai', 'sonata', 'sedan', '2010_2014'],
        15 : ['hyundai', 'sonata', 'sedan', '2019_2020'],
        16 : ['kia', 'carnival', 'van', '2015_2020'],
        17 : ['kia', 'carnival', 'van', '2021_'],
        18 : ['kia', 'k5', 'sedan', '2010_2015'],
        19 : ['kia', 'k5', 'sedan', '2020_'],
        20 : ['kia', 'k7', 'sedan', '2016_2020'],
        21 : ['kia', 'mohave', 'suv', '2020_'],
        22 : ['kia', 'morning', 'hatchback', '2004_2010'],
        23 : ['kia', 'morning', 'hatchback', '2011_2016'],
        24 : ['kia', 'ray', 'hatchback', '2012_2017'],
        25 : ['kia', 'sorrento', 'suv', '2015_2019'],
        26 : ['kia', 'sorrento', 'suv', '2020_'],
        27 : ['kia', 'soul', 'suv', '2014_2018'],
        28 : ['kia', 'sportage', 'suv', '2016_2020'],
        29 : ['kia', 'stonic', 'suv', '2017_2019'],
        30 : ['renault', 'sm3', 'sedan', '2015_2018'],
        31 : ['renault', 'xm3', 'suv', '2020_'],
        32 : ['ssangyong', 'korando', 'suv', '2019_2020'],
        33 : ['ssangyong', 'tivoli', 'suv', '2016_2020'],
    }
    
    df_datatag = pd.DataFrame.from_dict(json_datatag, orient='index', columns=['Brand', 'Model', 'Type', 'Year'])
    df_datatag = df_datatag.rename_axis('idx').reset_index()
    
    return json_datatag, df_datatag

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def ViT_dataloader(grouped_df):
    os.makedirs('data', exist_ok=True)
    
    train_dir = 'data/train'
    test_dir = 'data/val'
    
    result_train_list = []
    result_test_list = []
    result_train_label_list = []
    result_test_label_list = []
    
    for cls, index_list in grouped_df.items():
        
        cls_train_list = []
        cls_test_list = []
        
        for idx in index_list:
            train_list = glob.glob(os.path.join(train_dir, str(idx), '*.jpg'))
            test_list = glob.glob(os.path.join(test_dir, str(idx), '*.jpg'))
            cls_train_list.append(train_list)
            cls_test_list.append(test_list)
            
        flatten_cls_train_list = flatten_list(cls_train_list)
        flatten_cls_test_list = flatten_list(cls_test_list)
        tmp_train_labels = [cls] * len(flatten_cls_train_list)
        tmp_test_labels = [cls] * len(flatten_cls_test_list)
        
        result_train_list.append(flatten_cls_train_list)
        result_test_list.append(flatten_cls_test_list)
        result_train_label_list.append(tmp_train_labels)
        result_test_label_list.append(tmp_test_labels)
    
    flt_result_train_list = flatten_list(result_train_list)
    flt_result_test_list = flatten_list(result_test_list)
    flt_result_train_label_list = flatten_list(result_train_label_list)
    flt_result_test_label_list = flatten_list(result_test_label_list)
    
    return flt_result_train_list, flt_result_train_label_list, flt_result_test_list, flt_result_test_label_list

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, df_datatag, aim_class, grouped_df, transform=None):
        self.file_list = file_list
        self.df_datatag = df_datatag
        self.aim_class = aim_class
        self.grouped_df = grouped_df
        self.confirm_label = list(grouped_df.keys())
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        real_class = img_path.split("/")[-2]
        label = self.df_datatag[self.aim_class][int(real_class)]
        label = self.confirm_label.index(label)

        return img_transformed, label


def main():
    # print(f"Torch: {torch.__version__}")
    
    # Training settings
    batch_size = 1024
    epochs = 100
    lr = 3e-5
    gamma = 0.7
    seed = 42
    device = 'cuda'
    
    seed_everything(seed)
    
    json_datatag, df_datatag = make_data_tag()
    
    # Brand Classification
    aim_class = 'Brand'
    grouped_df = df_datatag.groupby(aim_class).apply(lambda x: x.index.tolist())
    train_list, labels, test_list, test_labels = ViT_dataloader(grouped_df)
    
    # random_idx = np.random.randint(1, len(train_list), size=9)
    
    # 동작 안함
    # fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # for idx, ax in enumerate(axes.ravel()):
    #     img = Image.open(train_list[idx])
    #     ax.set_title(labels[idx])
    #     ax.imshow(img)
    
    train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")
    
    # Image Augmentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    
    train_data = CatsDogsDataset(train_list, df_datatag, aim_class, grouped_df, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, df_datatag, aim_class, grouped_df, transform=test_transforms)
    test_data = CatsDogsDataset(  test_list, df_datatag, aim_class, grouped_df, transform=test_transforms)
    
    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    
    # print(len(train_data), len(train_loader))
    # print(len(valid_data), len(valid_loader))
    
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )
    
    model = ViT(
        dim=128,
        image_size=640,
        patch_size=32,
        num_classes=len(grouped_df.keys()),
        transformer=efficient_transformer,
        channels=3,
    ).to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
            
            file_path = 'model.pt'
            torch.save(model.state_dict(), file_path)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    

if __name__ == "__main__":
    main()
