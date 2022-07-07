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

    def __iter__(self):
        # initialize if not already initialized
        if self.len is None:
            self._init_db()
        # return a generator
        for i in range(self.len):
            yield self.__getitem__(i)

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

        return img, labels_onehot


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


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        all_batches = [b for b in self.dl]
        for batch in all_batches:
            device_batch = to_device(batch, self.device)
            print(device_batch[0].get_device(), device_batch[1].get_device())
            yield device_batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    @property
    def batch_size(self):
        return self.dl.batch_size


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
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                              num_workers=0, drop_last=True)

    val_loader = DataLoader(val_data, batch_size=32, shuffle=False,
                            num_workers=0, drop_last=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
    #                        num_workers=1, drop_last=True)

    device = get_default_device()
    # loading training and validation data onto GPU
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)
    test_dl = val_dl
    return train_dl, val_dl, test_dl, LABELS  # todo fix when we have labels for val and test


if __name__ == '__main__':
    dataloader = LMDBLoader("data/deepglobe_patches/train", None)
    for sample in dataloader:
        print(sample)

    tl, vl,test_loader, labels = load_data()
    for batch in tqdm(tl, mininterval=0.5, desc='(Training)', leave=False):
        print(batch)
        break
