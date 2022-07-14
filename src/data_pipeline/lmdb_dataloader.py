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
        print(f"LMDB Initialized with Len:{self.len}")

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

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        all_batches = [b for b in self.dl]
        for batch in all_batches:
            device_batch = to_device(batch, self.device)
            # print(device_batch[0].get_device(), device_batch[1].get_device())
            yield device_batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    @property
    def batch_size(self):
        return self.dl.batch_size


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


def load_data_from_lmdb(data_dir="/data/deepglobe_patches/", transformations=None, batch_size=64):
    """
    Pre-processing for our images
    Resizing because images have different sizes by default
    Converting each image from a numpy array to a tensor (so we can do calculations on the GPU)
    Normalizing the image as following: image = (image - mean) / std
    """

    # mean and std values of the Imagenet Dataset so that pretrained models could also be used
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # default transformations
    if not transformations:
        transformations = transforms.Compose([
            transforms.ToPILImage(),  # to PIL such that it can be converted to tensor
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomRotation(degrees=(-90, 90)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(*imagenet_stats)
        ])

    # Getting the data
    train_data = LMDBLoader(data_dir + "train", transformation=transformations)
    # val_data = LMDBLoader(data_dir + "valid", transform=transformations)
    # test_data = LMDBLoader(data_dir + "test", transform=transformations)

    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, test_size])

    # Create the dataloader for each dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
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
    dataloader = LMDBLoader("data/deepglobe_patches/train", additive_noise=0.02, subtractive_noise=0.04)
    dist = dataloader._get_label_indexes()
    print(dist)
