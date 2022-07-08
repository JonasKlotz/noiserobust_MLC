import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm.notebook import tqdm
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from data_pipeline.lmdb_dataloader import DeviceDataLoader
import os
import re
import requests

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
classes = ['black', 'blue', 'brown', 'green', 'white', 'red', 'dress', 'pants', 'shorts', 'shoes', 'shirt']
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# mean and std values of the Imagenet Dataset so that pretrained models could also be used


def encode_label(label, classes_list=classes):  # encoding the classes into a tensor of shape (11) with 0 and 1s.
    target = torch.zeros(11)
    for l in label:
        idx = classes_list.index(l)
        target[idx] = 1
    return target


def decode_target(target, threshold=0.5):  # decoding the prediction tensors of 0s and 1s into text form
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            result.append(classes[i])
    return ' '.join(result)


# A class to create a Custom Dataset that will load images and encode the labels of those images from their folder names
class myDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.images = get_path_names(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        label = re.findall(r'\w+\_\w+', img_path)[0].split('_')

        return img, encode_label(label)

    # Making a list that contains the paths of each image


def get_path_names(dir):
    images = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            # print(os.path.join(path, name))
            images.append(os.path.join(path, name))
    return images


def denorm(img_tensors):  # this function will denormalize the tensors
    return img_tensors * imagenet_stats[1][0] + imagenet_stats[0][0]


def show_example(img, label):
    plt.imshow(denorm(img).permute(1, 2, 0))
    print("Label:", decode_target(label))
    print()
    print(label)


# helper functions to load the data and model onto GPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def load_apparel_data(data_dir='data/apparel-images-dataset', batch_size = 32):
    # creating a list of classes

    # setting a set of transformations to transform the images
    transform = T.Compose([T.Resize(128),
                           T.RandomCrop(128),
                           T.RandomHorizontalFlip(),
                           T.RandomRotation(2),
                           T.ToTensor(),
                           T.Normalize(*imagenet_stats)])

    dataset = myDataset(data_dir, transform=transform)

    val_percent = int(0.15 * len(dataset))  # setting 15 percent of the total number of images for validation
    train_size = len(dataset) - val_percent
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset,[train_size, val_size])  # splitting the dataset for training and validation.

    # setting batch size for Dataloader to load the data batch by batch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=0, drop_last=True)

    device = get_default_device()

    # loading training and validation data onto GPU
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)
    test_dl = val_dl
    return train_dl, val_dl, test_dl, classes
