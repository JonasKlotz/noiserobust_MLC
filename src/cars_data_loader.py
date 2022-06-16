from os import listdir
import os

import torch
import urllib3
import shutil
import tarfile
from scipy import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms, models

download = False
data_dir = "../data/cars/"

def getting_data(url, path):
    file_name = "car_imgs.tgz"
    http = urllib3.PoolManager()
    with open(file_name, 'wb') as out:
        r = http.request('GET', url, preload_content=False)
        shutil.copyfileobj(r, out)

    tar_package = tarfile.open(file_name, mode='r:gz')
    tar_package.extractall(path)
    tar_package.close()
    return print("Data extracted and saved.")


def getting_metadata(url, file_name):
    '''
  Downloading a metadata file from a specific url and save it to the disc.
  '''
    http = urllib3.PoolManager()
    with open(file_name, 'wb') as out:
        r = http.request('GET', url, preload_content=False)
        shutil.copyfileobj(r, out)
    return print("Metadata downloaded and saved.")


class MetaParsing():
    '''
  Class for parsing image and meta-data for the Stanford car dataset to create a custom dataset.
  path: The filepah to the metadata in .mat format.
  *args: Accepts dictionaries with self-created labels which will be extracted from the metadata (e.g. {0: 'Audi', 1: 'BMW', 3: 'Other').
  year: Can be defined to create two classes (<=year and later).
  '''

    def __init__(self, path, *args, year=None):
        self.mat = io.loadmat(path)
        self.year = year
        self.args = args
        self.annotations = np.transpose(self.mat['annotations'])
        # Extracting the file name for each sample
        self.file_names = [annotation[0][0][0].split("/")[-1] for annotation in self.annotations]
        # Extracting the index of the label for each sample
        self.label_indices = [annotation[0][5][0][0] for annotation in self.annotations]
        # Extracting the car names as strings
        self.car_names = [x[0] for x in self.mat['class_names'][0]]
        # Create a list with car names instead of label indices for each sample
        self.translated_car_names = [self.car_names[x - 1] for x in self.label_indices]

    def brand_types(self, base_dict, x):
        y = list(base_dict.keys())[-1]
        for k, v in base_dict.items():
            if v in x: y = k
        return y

    def parsing(self):
        """
        @return:  a list containing three lists of numeric features for our three classes (brand, type, year).
                  These are our training labels

        """
        result = []
        for arg in self.args:
            temp_list = [self.brand_types(arg, x) for x in self.translated_car_names]
            result.append(temp_list)
        if self.year != None:
            years_list = [0 if int(x.split(" ")[-1]) <= self.year else 1 for x in self.translated_car_names]
            result.append(years_list)
        brands = [x.split(" ")[0] for x in self.translated_car_names]
        return result, self.file_names, self.translated_car_names


def count_classes(base_dict, base_list):
    for i in range(len(list(base_dict.keys()))):
        print("{}: {}".format(base_dict[i], str(base_list.count(i))))


class CarDataset(Dataset):

    def __init__(self, car_path, transform, translation_dict):
        self.path = car_path
        self.folder = [x for x in listdir(car_path)]
        self.transform = transform
        self.translation_dict = translation_dict
    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.path, self.folder[idx])
        image = Image.open(img_loc).convert('RGB')
        single_img = self.transform(image)

        label1 = torch.tensor([self.translation_dict[self.folder[idx]][0]])
        label2 = torch.tensor([self.translation_dict[self.folder[idx]][1]])
        label3 = torch.tensor([self.translation_dict[self.folder[idx]][2]])

        label_tensor = torch.stack((label1, label2, label3), dim=1)

        sample = {'image':single_img,
                  'labels': label_tensor}
        return sample


def load_cars_dataset():

    if download:
        getting_data("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz",
                     data_dir + "carimages")
        getting_metadata("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat",
                         data_dir + "car_metadata.mat")

    brand_dict = {0: 'Audi', 1: 'BMW', 2: 'Chevrolet', 3: 'Dodge', 4: 'Ford', 5: 'Other'}
    vehicle_types_dict = {0: 'Convertible', 1: 'Coupe', 2: 'SUV', 3: 'Van', 4: 'Other'}

    results, file_names, translated_car_names = MetaParsing(data_dir + "car_metadata.mat",
                                                            brand_dict, vehicle_types_dict, year=2009).parsing()


    count_classes(brand_dict, results[0])
    count_classes(vehicle_types_dict, results[1])

    translation_dict = dict(zip(file_names, list(zip(results[0], results[1], results[2]))))

    # Pre-processing for our images
    # Resizing because images have different sizes by default
    # Converting each image from a numpy array to a tensor (so we can do calculations on the GPU)
    # Normalizing the image as following: image = (image - mean) / std
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Getting the data
    cardata = CarDataset(data_dir + "car_ims", transform=data_transforms, translation_dict=translation_dict)

    # Split the data in training and testing
    train_len = int(cardata.__len__() * 0.8)
    test_len = int(cardata.__len__() * 0.2)
    train_set, val_set = torch.utils.data.random_split(cardata, [train_len, test_len])

    # Create the dataloader for each dataset
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=1, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False,
                            num_workers=1, drop_last=True)

    """sample = next(iter(train_loader))

    print("Keys in our sample batch: {}".format(sample.keys()))
    print("Size for the images in our sample batch: {}".format(sample['image'].shape))
    print("Size for the target in our sample batch: {}".format(sample['labels']['label_brand'].shape))
    print("Targets for each batch in our sample: {}".format(sample['labels']['label_brand']))"""

    return train_loader, val_loader
