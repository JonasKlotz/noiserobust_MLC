import random
import time

from matplotlib import pyplot as plt
import os
import numpy as np
import lmdb
import pickle

np.random.seed(42)

LABELS = {
    '[0. 1. 1.]': 'urban_land',
    '[1. 1. 0.]': 'agriculture_land',
    '[1. 0. 1.]': 'rangeland',
    '[0. 1. 0.]': 'forest_land',
    '[0. 0. 1.]': 'water',
    '[1. 1. 1.]': 'barren_land',
    '[0. 0. 0.]': 'unknown'
}


class Subsampler:
    """ Class that is used to create subsamples from a single image, and extract the labels associated """

    def __init__(self, img, img_labels, patch_size=256, patches=20):
        self.img = img
        self.labels_img = img_labels
        self.size = patch_size
        self.patches = patches

        # assert self.labels_img is None or self.labels_img.shape[0] == self.img.shape[0]

    def get_all_subsamples_from_img(self):
        """
        Creates multiple subsamples from a given image. This is achived by a random sampling approach until finished.

        :return: list of subsamples
        """

        subsamples_list = []
        # img_labels = self.labels_img.reshape(-1, 3)
        dominating_label = [1., 1., 0.]
        # indexes = np.where(img_labels != dominating_label)

        # iterate until the number of correct subsamples is reached
        while len(subsamples_list) < self.patches:
            x = int((self.img.shape[0] - self.size) * np.random.random())
            y = int((self.img.shape[1] - self.size) * np.random.random())

            # check if the subsample would overlap with an already existing subsample
            is_overlap = False
            for previous in subsamples_list:
                is_x_overlap = previous['x'] - self.size <= x <= previous['x'] + self.size
                is_y_overlap = previous['y'] - self.size <= y <= previous['y'] + self.size
                if is_x_overlap and is_y_overlap:
                    is_overlap = True
            if is_overlap:
                continue

            # if the subsample is valid, add it to the list
            subsample = self._get_single_subsample(x, y)
            subsamples_list.append(subsample)

        return subsamples_list

    def _get_single_subsample(self, x, y):
        """
        Creates a subsample from the class image.

        :param x: x coordinate of the start of the subsample
        :param y: y coordinate of the start of the subsample
        :return: subsample
        """

        assert self.img.shape[0] >= self.size + x, "Invalid x coordinate"
        assert self.img.shape[1] >= self.size + y, "Invalid y coordinate"

        subsample = {'x': x,
                     'y': y,
                     'size': self.size,
                     'img': self.img[x:x + self.size, y:y + self.size]}

        # if labels are available, get the labels from the subsample
        if self.labels_img is not None:
            subsample_label_img = self.labels_img[x:x + self.size, y:y + self.size]
            subsample['labels'] = self.get_labels_from_label_img(subsample_label_img)

        return subsample

    @staticmethod
    def get_labels_from_label_img(label_img, min_percentage=0.03):
        """
        Searches through the label image and returns the names of all labels present.

        returns: list of label names
        """

        # reshape the label image to a 1D array of pixels
        label_pixels = label_img.reshape(-1, 3)

        # convert the label pixels to a list of labels
        label_counts = np.unique(label_pixels, return_counts=True, axis=0)
        min_limit = int(label_pixels.shape[0] * min_percentage)
        # get the counts of each label, where a label is defined as a pixel with three integer values
        label_counts = {LABELS[str(label_count)]: count for label_count, count in zip(label_counts[0], label_counts[1]) if count > min_limit}

        # remove the 'unknown' label
        label_counts.pop('unknown', None)

        # convert to list of labels
        labels = list(label_counts.keys())

        return labels


# TODO: agriculture_land_count: 6749 total: 9636
def subsample_whole_dir(dir_path):
    """
    Subsamples all images in a directory (deepglobe/train etc.) and saves all subsamples to one lmdb file.

    :param dir_path: path to the image directory
    """

    # iterate over all images in the directory that are satelite .jpg images
    all_subsamples = []
    all_labels = []
    images = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]

    for subsample_index, img_filename in enumerate(images):

        print(subsample_index, img_filename)  # debug

        # get full paths for loading the original and the labels image
        path = dir_path + "/" + img_filename
        path_labels = path.replace('sat.jpg', 'mask.png')

        # create subsampler Class to subsample the image and get the subsamples (including labels if available)
        subsampler = Subsampler(img=plt.imread(path), img_labels=plt.imread(path_labels))
        subsamples = subsampler.get_all_subsamples_from_img()
        all_subsamples.extend(subsamples)

        all_labels += [subsample['labels'] for subsample in subsamples]

    # get the percentage of 'agriculture_land' in the labels
    agriculture_land_count = sum([1 for label in all_labels if 'agriculture_land' in label])
    print("agriculture_land_count:", agriculture_land_count, 'total:', len(all_labels))

    return all_subsamples


def remove_only_aggreculture_subsamples(subsamples, percentage = 0.4, split_negative_ratio = 0.7):
    """ Function that removes percentage of all subsamples that only contain aggrecultural land label """

    # remove the percentage of subsamples that only contain agriculture land label
    subsamples_positive = [subsample for subsample in subsamples if 'agriculture_land' not in subsample['labels']]
    subsamples_negative = [subsample for subsample in subsamples if 'agriculture_land' in subsample['labels'] and len(subsample['labels']) == 1]
    subsamples_neutral = [subsample for subsample in subsamples if 'agriculture_land' in subsample['labels'] and len(subsample['labels']) > 1]

    # get the percentage of subsamples that only contain agriculture land label
    amount_to_remove_negative = int(percentage * len(subsamples) * split_negative_ratio)
    amount_to_remove_neutral = int((1-split_negative_ratio) * percentage * len(subsamples))
    subsamples_negative_shortened = subsamples_negative[:-amount_to_remove_negative]
    subsamples_neutral_shortened = subsamples_neutral[:-amount_to_remove_neutral]

    # combine the subsamples
    subsamples_shortened = subsamples_positive + subsamples_negative_shortened + subsamples_neutral_shortened

    print("Aggrecultural Land count shortened:", len(subsamples_negative_shortened)+len(subsamples_neutral_shortened))

    return subsamples_shortened


def save_subsamples_to_lmdb(subsamples, dir_path, splits):

    # shuffle the subsamples
    random.shuffle(subsamples)

    # split the subsamples into training and validation subsamples
    subsamples_train = subsamples[:int(len(subsamples) * splits['train'])]
    subsamples_val = subsamples[int(len(subsamples) * splits['train']):int(len(subsamples) * (splits['train'] + splits['val']))]
    subsamples_test = subsamples[int(len(subsamples) * (splits['train'] + splits['val'])):]
    subsamples_dict = {'train': subsamples_train, 'val': subsamples_val, 'test': subsamples_test}

    for split_name, split in splits.items():
        path = dir_path + "/" + split_name
        if not os.path.exists(path):
            os.makedirs(path)

        # open the lmdb environment, with enough space for all subsamples
        map_size = int((983298 + 1000) * len(subsamples_dict[split_name]) * split * 4)
        env = lmdb.open(path, map_size=map_size)

        # iterate over subsamples and save them as key/value pair to the lmdb file, where the key is an overall index
        for subsample_index, subsample in enumerate(subsamples_dict[split_name]):
            key = f"{subsample_index}"

            # save the subsample as byte array to the lmdb file
            with env.begin(write=True) as txn:
                key_bytes = key.encode('ascii')
                value_bytes = pickle.dumps(subsample)
                txn.put(key_bytes, value_bytes)
        env.close()



def try_lmdb(path='data/deepglobe_patches/train'):
    """
    Test the lmdb file by loading it and printing the first subsample.
    """

    # load the lmdb file
    env = lmdb.open(path)
    txn = env.begin()
    # print the length of the lmdb file
    print("Length:", txn.stat()['entries'])
    # get the keys of the lmdb file decoded
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    print(keys[0])
    # get a single subsample
    values = [txn.get(key.encode()) for key in keys]
    print(values[0])
    values = [pickle.loads(value) for value in values]
    print(values[0])
    labels = [value['labels'] for value in values]
    print(labels[0])

    # store the labels as a text file
    with open("results/data/deepglobe-patches_labels.txt", 'w') as f:
        for labels in labels:
            labels_str = ','.join(labels)
            f.write(str(labels_str) + '\n')


if __name__ == "__main__":
    subsamples = subsample_whole_dir(dir_path="data/deepglobe/train")
    subsamples_modified = remove_only_aggreculture_subsamples(subsamples)
    save_subsamples_to_lmdb(subsamples_modified,
                            dir_path="data/deepglobe_patches",
                            splits={'train': 0.7, 'val': 0.1, 'test': 0.2})
