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
        self.keys = [key for key, _ in self.txn.cursor()]

    def __len__(self):
        if self.len is None:
            self._init_db()
        return self.len

    def __getitem__(self, idx):
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()

        key = self.keys[idx]
        value = self.txn.get(key, readonly=True, buffers=True)

        img = value['img']
        img = self.transform(img)
        labels = value['labels']

        return {'image': img, 'labels': labels}


def load_data(data_dir="data/deepglobe_patches/", transformations=None):
    # Pre-processing for our images
    # Resizing because images have different sizes by default
    # Converting each image from a numpy array to a tensor (so we can do calculations on the GPU)
    # Normalizing the image as following: image = (image - mean) / std
    if not transformations:
        transformations = transforms.Compose([
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
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=1, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False,
                            num_workers=1, drop_last=True)

    return train_loader, val_loader, LABELS


if __name__ == '__main__':
    tl, vl, labels = load_data()
    for batch in tqdm(tl, mininterval=0.5, desc='(Training)', leave=False):
        print(batch)
        break
