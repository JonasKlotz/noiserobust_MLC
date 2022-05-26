import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
from utils import util
import glob
from collections import defaultdict

object_categories = []
fn_map = {}
for fn in glob.glob("/mnt/beegfs/bulk/mirror/wz346/noisy_mlc/data/nuswide/images/*.jpg"):
    tmp = fn.split('/')[-1].split('_')[1]
    fn_map[tmp] = fn

def read_info(root, set):
    imagelist = {}
    hash2ids = {}
    if set == "trainval": 
        path = os.path.join(root, "ImageList", "TrainImagelist.txt")
    elif set == "test":
        path = os.path.join(root, "ImageList", "TestImagelist.txt")
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split('\\')[-1]
            start = line.index('_')
            end = line.index('.')
            imagelist[i] = line[start+1:end]
            hash2ids[line[start+1:end]] = i

    return imagelist

def read_object_labels(root, dataset, set, imagelist, noisy_label='none', noisy_level=0):
    if set == "trainval":
        label_files = glob.glob(root+"TrainTestLabels/Labels_*Train.txt")
    elif set == "test":
        label_files = glob.glob(root+"TrainTestLabels/Labels_*Test.txt")

    if set == "trainval":
        for fn in label_files:
            label = fn.split('/')[-1].split('_')[1] 
            object_categories.append(label)
    elif set == "test":
        label_files = []
        for item in object_categories:
            label_files.append(root+"TrainTestLabels/Labels_{}_Test.txt".format(item))

    ids2labels = defaultdict(list)
    for fn in label_files:
        with open(fn, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                line = (-1 if line == '0' else 1)
                ids2labels[i].append(line)

    # remove the labels if the image doesn't exist any more
    removed = []
    for key in ids2labels:
        if imagelist[key] not in fn_map:
            removed.append(key)
    for item in removed:
        del ids2labels[item]
        del imagelist[item]

    prev_mean = 0
    mean = 0
    if set == 'trainval':
        if noisy_label == 'uniform':
            for key in ids2labels:
                prev_mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
                for i in range(len(ids2labels[key])):
                    rnd = np.random.rand()
                    if rnd < noisy_level:
                        ids2labels[key][i] *= -1
                mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
        elif noisy_label == 'uniform-positive':
            for key in ids2labels:
                prev_mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
                for i in range(len(ids2labels[key])):
                    rnd = np.random.rand()
                    if rnd < noisy_level and ids2labels[key][i] == 1:
                        ids2labels[key][i] = -1
                mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
        elif noisy_label == 'one-positive':
            for key in ids2labels:
                prev_mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
                ones = []
                for i in range(len(ids2labels[key])):
                    if ids2labels[key][i] == 1:
                        ones.append(i)
                if len(ones) > 1:
                    choice = np.random.choice(ones)
                    for x in ones:
                        if x != choice:
                            ids2labels[key][x] = -1 
                mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
        elif noisy_label == 'combined':
            for key in ids2labels:
                rnd1 = np.random.rand()
                prev_mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
                if rnd1 < 1/3:
                    for i in range(len(ids2labels[key])):
                        rnd = np.random.rand() + 0.5
                        if rnd < noisy_level:
                            ids2labels[key][i] *= -1
                elif 1/3 <= rnd1 < 2/3:
                    for i in range(len(ids2labels[key])):
                        rnd = np.random.rand() + 0.5
                        if rnd < noisy_level and ids2labels[key][i] == 1:
                            ids2labels[key][i] = -1
                else:
                    ones = []
                    for i in range(len(ids2labels[key])):
                        if ids2labels[key][i] == 1:
                            ones.append(i)
                    if len(ones) > 1:
                        choice = np.random.choice(ones)
                        for x in ones:
                            if x != choice:
                                ids2labels[key][x] = -1 
                mean += np.sum([1 for xx in ids2labels[key] if xx == 1])
                    
            
    print("ori mean:", prev_mean/len(ids2labels))
    print("now mean:", mean/len(ids2labels))
    return ids2labels 


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(len(object_categories)):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, imagelist, fn_map, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = int(row[0])
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = np.where(labels<0, 0.0, 1.0)
                labels = torch.from_numpy(labels)
                name2 = fn_map[imagelist[name]]
                item = (name2, labels)
                images.append(item)
            rownum += 1
    return images


class NUSWIDEClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None, noisy_label=False, noisy_level=0):
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'nuswide')
        if noisy_label != 'none' and set == 'trainval': 
            file_csv = os.path.join(path_csv, 'classification_' + set + 'noisy.' + noisy_label + str(noisy_level).split('.')[-1] + '.csv')
        else:
            # define filename of csv file
            file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        imagelist = read_info(root, set)
        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'nuswide', self.set, imagelist, noisy_label, noisy_level)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv, imagelist, fn_map)

        print('[dataset] NUSWIDE classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, 0), 0, (target, 0)

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
