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
#import util
from itertools import combinations

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


# noisy_label = {'none', 'uniform', 'uniform-positive', 'one-positive'}
def read_image_label(file, set):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, set, noisy_label='none', noisy_level=0):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file, set)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    prev_mean = 0
    mean = 0
    if set == 'trainval':
        if noisy_label == 'uniform':
            for key in labeled_data:
                prev_mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
                for i in range(len(labeled_data[key])):
                    rnd = np.random.rand()
                    if rnd < noisy_level:
                        labeled_data[key][i] *= -1
                mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
        elif noisy_label == 'uniform-positive':
            for key in labeled_data:
                prev_mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
                for i in range(len(labeled_data[key])):
                    rnd = np.random.rand()
                    if rnd < noisy_level and labeled_data[key][i] == 1:
                        labeled_data[key][i] = -1
                mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
        elif noisy_label == 'one-positive':
            for key in labeled_data:
                prev_mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
                ones = []
                for i in range(len(labeled_data[key])):
                    if labeled_data[key][i] == 1:
                        ones.append(i)
                choice = np.random.choice(ones)
                for x in ones:
                    if x != choice:
                        labeled_data[key][x] = -1 
                mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
        elif noisy_label == 'combined':
            for key in labeled_data:
                prev_mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
                rnd1 = np.random.rand()
                for i in range(len(labeled_data[key])):
                    if rnd1 < 1/3:
                        rnd = np.random.rand() + 0.5
                        if rnd < noisy_level:
                            labeled_data[key][i] *= -1
                    elif 1/3 <= rnd1 < 2/3:
                        rnd = np.random.rand() + 0.5
                        if rnd < noisy_level and labeled_data[key][i] == 1:
                            labeled_data[key][i] = -1
                    else:
                        ones = []
                        for i in range(len(labeled_data[key])):
                            if labeled_data[key][i] == 1:
                                ones.append(i)
                        choice = np.random.choice(ones)
                        for x in ones:
                            if x != choice:
                                labeled_data[key][x] = -1 
                mean += np.sum([1 for xx in labeled_data[key] if xx == 1])
        else:
            print(noisy_label, "not supported")

    print("ori mean:", prev_mean/len(labeled_data))
    print("now mean:", mean/len(labeled_data))
    return labeled_data


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
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
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
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = np.where(labels<0, 0.0, 1.0)
                #labels = [int(xx) for xx in row[1:num_categories + 1]]
                #labels = [2]+[i+4 for i in range(len(labels)) if labels[i] > 0]+[3]
                #labels = np.asarray(labels)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc2007(root):
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
            util.download_url(urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['trainval_2007'], cached_file))
            util.download_url(urls['trainval_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_images_2007'], cached_file))
            util.download_url(urls['test_images_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
            util.download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


class Voc2007Classification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, adj=None, noisy_label='none', noisy_level=0):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        if noisy_label != 'none' and set == 'trainval': 
            file_csv = os.path.join(path_csv, 'classification_' + set + 'noisy.' + noisy_label + str(noisy_level).split('.')[-1] + '.csv')
        else:
            # define filename of csv file
            file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.set, noisy_label, noisy_level)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        if set == 'trainval':
            self.adj = {'nums': [0 for _ in range(len(object_categories))], 'adj': [[0 for _ in range(len(object_categories))] for _ in range(len(object_categories))]}
            for item in self.images:
                for i in range(len(item[1])):
                    if item[1][i] == 1:
                        self.adj['nums'][i] += 1
            for item in self.images:
                tmp = [j for j in range(len(item[1])) if item[1][j] == 1]
                for u, v in combinations(tmp, 2):
                    self.adj['adj'][u][v] += 1
                    self.adj['adj'][v][u] += 1
            self.adj['nums'] = np.array(self.adj['nums'])
            self.adj['adj'] = np.array(self.adj['adj'])

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, 0), 0, (target, 0)

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
