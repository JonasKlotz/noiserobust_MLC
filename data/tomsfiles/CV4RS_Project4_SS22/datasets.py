import csv
import lmdb
import numpy as np
from skimage.transform import resize
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import read_json


class Ben19Dataset(Dataset):
    def __init__(self, lmdb_path, csv_path, transform_mode='default', active_classes=None, nomenclature=19):
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True)
        self.patch_names = self.read_csv(csv_path)
        self.active_classes = active_classes
        self.nomenclature = nomenclature
        self.transform = self.init_transform(transform_mode)

    def read_csv(self, csv_data):
        patch_names = []
        with open(csv_data, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                patch_names.append(row[0])
        return patch_names

    def init_transform(self, transform_mode):
        if transform_mode == 'default':
            return transforms.Compose([ToTensorBEN19(), NormalizeBEN19()])
        else:
            return self.transform_mode

    def interpolate_bands(self, bands, img10_shape=[120, 120]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp
    
    def convert_to_multihot(self, labels):
        d = read_json('/workspace/temp/label_mapping.json')
        multihot = np.zeros(19)
        indices = [d['ben19_name2idx'][label] for label in labels]
        multihot[indices] = 1
        
        if self.active_classes is not None:
            multihot = multihot[self.active_classes]
            
        return multihot

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        patch_name = self.patch_names[idx]
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))
            s2_patch = BigEarthNet_S2_Patch.loads(byteflow)

        metadata = s2_patch.__stored_args__
        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)
        label = self.convert_to_multihot(metadata['new_labels'])
        
        sample = dict(bands10=bands10,
                      bands20=bands20,
                      label=label,
                      index=idx)
        sample = self.transform(sample)

        return sample

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


class NormalizeBEN19(object):
    """BEN19 specific normalization."""

    def __init__(self):

        self.s2_bands10_mean = torch.Tensor([429.9430203, 614.21682446, 590.23569706, 2218.94553375])
        self.s2_bands10_std = torch.Tensor([572.41639287, 582.87945694, 675.88746967, 1365.45589904])

        self.s2_bands20_mean = torch.Tensor([950.68368468, 1792.46290469, 2075.46795189,
                                             2266.46036911, 1594.42694882, 1009.32729131])
        self.s2_bands20_std = torch.Tensor([729.89827633, 1096.01480586, 1273.45393088,
                                            1356.13789355, 1079.19066363, 818.86747235])

        self.s2_bands60_mean = torch.Tensor([340.76769064, 2246.0605464])
        self.s2_bands60_std = torch.Tensor([554.81258967, 1302.3292881])

        self.normalize_bands10 = transforms.Normalize(self.s2_bands10_mean, self.s2_bands10_std)
        self.normalize_bands20 = transforms.Normalize(self.s2_bands20_mean, self.s2_bands20_std)
        self.normalize_bands60 = transforms.Normalize(self.s2_bands60_mean, self.s2_bands60_std)

    def __call__(self, sample):

        bands10 = sample['bands10']
        bands20 = sample['bands20']
        label = sample['label']
        index = sample['index']
        bands10 = self.normalize_bands10(bands10)
        bands20 = self.normalize_bands20(bands20)
        img = torch.cat((bands10, bands20), dim=0)

        return dict(img=img,
                    label=label,
                    index=index)


class ToTensorBEN19(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return dict(bands10=torch.tensor(sample['bands10']),
                    bands20=torch.tensor(sample['bands20']),
                    label=sample['label'],
                    index=sample['index'])
