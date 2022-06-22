import pickle

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import lmdb
import os
from tqdm import tqdm

LABELS = [
    'urban_land',
    'agriculture_land',
    'rangeland',
    'forest_land',
    'water',
    'barren_land',
    'unknown'
]


class LMDBLoader(Dataset):

    def __init__(self, path, transform):
        self.path = path
        self.env = None
        self.keys = None
        self.len = None
        self.transform = transform

    def _init_db(self):
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()
        # get all keys and the overall length of the dataset
        self.len = self.txn.stat()['entries']
        self.keys = [key.decode('ascii') for key, _ in self.txn.cursor()]
        print(f"LMDB Initialized with Len:{self.len}")

    def __len__(self):
        if self.len is None:
            self._init_db()
        return self.len

    def __getitem__(self, idx):
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()

        key = self.keys[idx]
        #print(key)
        key_unicode = key.encode()
        value_data = self.txn.get(key_unicode)
        value = pickle.loads(value_data)

        img = value['img']
        if self.transform:
            img = self.transform(img)

        # transform image to tensor
        #img = torch.from_numpy(img) -> done in transformation

        # get array of onehot encoded labels
        labels = value['labels']
        labels_onehot = np.zeros(len(LABELS), dtype=np.int64)
        for label in labels:
            labels_onehot[LABELS.index(label)] = 1
        labels_onehot = torch.from_numpy(labels_onehot)

        return {'image': img, 'labels': labels_onehot, 'labels_string': labels}


def load_data(data_dir="/data/deepglobe_patches/", transformations=None):
    # Pre-processing for our images
    # Resizing because images have different sizes by default
    # Converting each image from a numpy array to a tensor (so we can do calculations on the GPU)
    # Normalizing the image as following: image = (image - mean) / std
    if not transformations:
        transformations = transforms.Compose([
            transforms.ToPILImage(), # to PIL such that it can be converted to tensor
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Getting the data
    trainloader = LMDBLoader(data_dir+"train", transform=transformations)

    # Split the data in training and testing
    train_len = int(np.ceil(len(trainloader) * 0.8))
    valid_len = int(np.floor(len(trainloader) * 0.2))

    train_set, val_set = torch.utils.data.random_split(trainloader, [train_len, valid_len])

    # Create the dataloader for each dataset
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=1, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=1, drop_last=True)

    return train_loader, val_loader, LABELS


if __name__ == '__main__':
    dataloader = LMDBLoader("data/deepglobe_patches/train", None)
    for sample in dataloader:
        print(sample)



    tl, vl, labels = load_data()
    for batch in tqdm(tl, mininterval=0.5, desc='(Training)', leave=False):
        print(batch)
        break
