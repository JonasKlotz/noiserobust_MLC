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
        # print(key)
        key_unicode = key.encode()
        value_data = self.txn.get(key_unicode)
        value = pickle.loads(value_data)

        img = value['img']
        if self.transform:
            img = self.transform(img)

        # transform image to tensor
        # img = torch.from_numpy(img) -> done in transformation

        # get array of onehot encoded labels
        labels_onehot = np.zeros(len(LABELS), dtype=np.int64)
        try:
            labels = value['labels']
            for label in labels:
                labels_onehot[LABELS.index(label)] = 1
        except KeyError:
            print("no labels given")

        labels_onehot = torch.from_numpy(labels_onehot)

        return {'image': img, 'labels': labels_onehot, 'labels_string': LABELS}


def load_data(data_dir="/data/deepglobe_patches/", transformations=None):
    # Pre-processing for our images
    # Resizing because images have different sizes by default
    # Converting each image from a numpy array to a tensor (so we can do calculations on the GPU)
    # Normalizing the image as following: image = (image - mean) / std
    if not transformations:
        transformations = transforms.Compose([
            transforms.ToPILImage(),  # to PIL such that it can be converted to tensor
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.RandomVerticalFlip(p=0.3),
            # transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomRotation(degrees=(-90, 90)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Getting the data
    train_data = LMDBLoader(data_dir + "train", transform=transformations)
    # val_data = LMDBLoader(data_dir + "valid", transform=transformations)
    # test_data = LMDBLoader(data_dir + "test", transform=transformations)

    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, test_size])
    # Create the dataloader for each dataset
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,
                              num_workers=1, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=1, drop_last=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
    #                        num_workers=1, drop_last=True)

    return train_loader, val_loader, val_loader, LABELS  # todo fix when we have labels for val and test


if __name__ == '__main__':
    dataloader = LMDBLoader("data/deepglobe_patches/train", None)
    for sample in dataloader:
        print(sample)

    tl, vl, labels = load_data()
    for batch in tqdm(tl, mininterval=0.5, desc='(Training)', leave=False):
        print(batch)
        break
