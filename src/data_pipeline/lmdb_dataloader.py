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

    def __init__(self, path, transformation=None, additive_noise=0., subtractive_noise=0.):
        self.transform = transformation
        self.add_noise = additive_noise
        self.sub_noise = subtractive_noise
        self.path = path
        self.env = None
        self.keys = None
        self.len = None
        self.label_indexes = {'urban_land': [], 'agriculture_land': [], 'rangeland': [], 'forest_land': [], 'water': [],
                              'barren_land': [], 'unknown': []}
        self.add_noise_indexes = {'urban_land': [], 'agriculture_land': [], 'rangeland': [], 'forest_land': [], 'water': [],
                                  'barren_land': [], 'unknown': []}
        self.sub_noise_indexes = {'urban_land': [], 'agriculture_land': [], 'rangeland': [], 'forest_land': [], 'water': [],
                                  'barren_land': [], 'unknown': []}
        np.random.seed(0)  # fix random seed

    def _init_db(self):
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()
        # get all keys and the overall length of the dataset
        self.len = self.txn.stat()['entries']
        self.keys = [key.decode('ascii') for key, _ in self.txn.cursor()]
        print(f"LMDB Initialized - Len: {self.len} - NOISE: {self.add_noise}/{self.sub_noise}")

    def _get_label_indexes(self):
        """ Function that returns all present indexes for every label in the dataset and the indexes for the noise """

        if self.env is None:
            self._init_db()

        img_index = 0
        for key, value in self.txn.cursor():
            value = pickle.loads(value)
            labels = value['labels']
            for label in labels:
                self.label_indexes[label] += [img_index]
            img_index += 1

        # also calculate a list of indexes that are to be noised
        for label in LABELS:
            # get the indexes for the label
            present_indexes = self.label_indexes[label]
            missing_indexes = [i for i in range(self.len) if i not in present_indexes]

            # get the indexes for the noise
            add_noise_total = int(self.add_noise * len(present_indexes))
            sub_noise_total = int(self.sub_noise * len(present_indexes))
            self.add_noise_indexes[label] = set(np.random.choice(missing_indexes, add_noise_total, replace=False))
            self.sub_noise_indexes[label] = set(np.random.choice(present_indexes, sub_noise_total, replace=False))

    def _noisify_label(self, onehot_labels, index):
        """ Function that adds noise to a single images labels array (one-hot), using class attributes """

        if self.label_indexes is None:
            self._get_label_indexes()

        # iterate over all labels in the array
        for i, label_value in enumerate(onehot_labels):
            label_name = LABELS[i]
            is_subtractive_noise = index in self.sub_noise_indexes[label_name]
            is_additive_noise = index in self.add_noise_indexes[label_name]
            if label_value == 1 and is_subtractive_noise:
                onehot_labels[i] = 0
            elif label_value == 0 and is_additive_noise:
                onehot_labels[i] = 1

        return onehot_labels

    def __len__(self):
        if self.len is None:
            self._init_db()
        return self.len

    def __iter__(self):
        for i in range(self.len):
            yield self.__getitem__(i)  # return a generator

    def __getitem__(self, idx):

        # delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()

        # extract the data from the LMDB
        key = self.keys[idx]
        key_unicode = key.encode()
        value_data = self.txn.get(key_unicode)
        value = pickle.loads(value_data)
        img = value['img']

        # transform the data if wanted
        if self.transform:
            img = self.transform(img)

        # get labels as onehot encoded array
        labels_onehot = np.zeros(len(LABELS), dtype=np.int64)
        try:
            labels = value['labels']
            for label in labels:
                labels_onehot[LABELS.index(label)] = 1
        except KeyError:
            print("no labels given")

        # add noise to the labels
        if self.add_noise or self.sub_noise:
            labels_onehot = self._noisify_label(labels_onehot, idx)

        labels_onehot = torch.from_numpy(labels_onehot)  # convert to tensor

        return img, labels_onehot


class DeviceDataLoader:
    """ Wraps a dataloader to move data to a device """

    def __init__(self, dl, device=None):
        self.dl = dl
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        all_batches = [b for b in self.dl]
        for batch in all_batches:
            device_batch = self.to_device(batch, self.device)
            # print(device_batch[0].get_device(), device_batch[1].get_device())
            yield device_batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    @property
    def batch_size(self):
        return self.dl.batch_size

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


def load_data_from_lmdb(data_dir="data/deepglobe_patches/", transformations=None, batch_size=64, add_noise=0.0, sub_noise=0.0):
    """ Pre-processes images, including transformations like resizing and normalization """

    # mean and std values of the Imagenet Dataset for other datasets
    # imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    deepglobe_stats = ([0.4085, 0.3795, 0.2823], [0.1447, 0.1123, 0.1023]) # manually calculated - data_evaluation.py

    # default transformations
    if not transformations:
        transformations = transforms.Compose([
            transforms.ToPILImage(),  # to PIL such that it can be converted to tensor
            transforms.RandAugment(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(*deepglobe_stats)
        ])

    # Getting the data
    train_data = LMDBLoader(data_dir + "train", transformation=transformations, additive_noise=add_noise, subtractive_noise=sub_noise)
    val_data = LMDBLoader(data_dir + "val", transformation=transformations)
    test_data = LMDBLoader(data_dir + "test", transformation=transformations)

    # Create the dataloader for each dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # loading training and validation data onto GPU
    train_dl = DeviceDataLoader(train_loader)
    val_dl = DeviceDataLoader(val_loader)
    test_dl = DeviceDataLoader(test_loader)

    return train_dl, val_dl, test_dl, LABELS


if __name__ == '__main__':
    dataloader = LMDBLoader("data/deepglobe_patches/train", additive_noise=0.02, subtractive_noise=0.04)
    dist = dataloader._get_label_indexes()
    print(dist)
