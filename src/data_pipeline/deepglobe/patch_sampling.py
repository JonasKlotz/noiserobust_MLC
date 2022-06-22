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

    def __init__(self, img, img_labels, patch_size=256, patches=12):
        self.img = img
        self.labels_img = img_labels
        self.size = patch_size
        self.patches = patches

        # assert self.labels_img is None or self.labels_img.shape[0] == self.img.shape[0]

    def get_img_subsamples(self):
        """
        Creates multiple subsamples from a given image. This is achived by a random sampling approach until finished.

        :return: list of subsamples
        """

        subsamples_list = []

        # iterate until the number of correct subsamples is reached
        while len(subsamples_list) < self.patches:
            x = int((self.img.shape[0] - self.size) * np.random.random())
            y = int((self.img.shape[1] - self.size) * np.random.random())

            # check if the subsample would overlap with an already existing subsample
            for previous in subsamples_list:
                is_x_overlap = previous['x'] - self.size <= x <= previous['x'] + self.size
                is_y_overlap = previous['y'] - self.size <= y <= previous['y'] + self.size
                if is_x_overlap and is_y_overlap:
                    continue

            # if the subsample is valid, add it to the list
            try:
                subsample = self._get_single_subsample(x, y)
                subsamples_list.append(subsample)
            except:
                continue

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
    def get_labels_from_label_img(label_img):
        """
        Searches through the label image and returns the names of all labels present.

        returns: list of label names
        """

        # reshape the label image to a 1D array of pixels
        label_pixels = label_img.reshape(-1, 3)
        unique_pixels = np.unique(label_pixels, axis=0)

        # only keep unique labels from the list
        label_pixels_str = [str(pixel) for pixel in unique_pixels]
        labels = list(set(label_pixels_str))

        # map the labels to their names
        label_names = [LABELS[str(label)] for label in labels]

        return label_names


# TODO: agriculture_land_count: 6749 total: 9636
def subsample_whole_dir(dir_path):
    """
    Subsamples all images in a directory (deepglobe/train etc.) and saves all subsamples to one lmdb file.

    :param dir_path: path to the image directory
    """

    # set the labels present flag if we work on the training set
    labels_present = ('train' in dir_path)
    print("Files:", len(os.listdir(dir_path)))

    # create a path to save the lbmdb file to, also inside /data
    save_path = dir_path.replace('deepglobe', 'deepglobe_patches')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # open the lmdb environment, with enough space for all subsamples
    map_size = (983298 + 1000) * len(os.listdir(dir_path)) * 7
    env = lmdb.open(save_path, map_size=map_size)

    # create a list for label analysis
    all_labels = []

    # iterate over all images in the directory that are satelite .jpg images
    images = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
    for subsample_index, img_filename in enumerate(images):

        print(subsample_index, img_filename)  # debug

        # get full paths for loading the original and the labels image
        path = dir_path + "/" + img_filename
        path_labels = path.replace('sat.jpg', 'mask.png')

        # load image, and labels if exist
        img = plt.imread(path)
        if labels_present:
            img_labels = plt.imread(path_labels)
        else:
            img_labels = None

        # create subsampler Class to subsample the image and get the subsamples (including labels if available)
        subsampler = Subsampler(img=img, img_labels=img_labels)
        subsamples = subsampler.get_img_subsamples()

        if labels_present:
            subsample_labels = [subsample['labels'] for subsample in subsamples]
            all_labels += subsample_labels

        # iterate over subsamples and save them as key/value pair to the lmdb file, where the key is an overall index
        for subsample_index, subsample in enumerate(subsamples):
            key = f"{img_filename}_{subsample_index}"

            # save the subsample as byte array to the lmdb file
            with env.begin(write=True) as txn:
                key_bytes = key.encode('ascii')
                value_bytes = pickle.dumps(subsample)
                txn.put(key_bytes, value_bytes)

    env.close()  # close the lmdb file

    # get the percentage of 'agriculture_land' in the labels
    agriculture_land_count = sum([1 for label in all_labels if 'agriculture_land' in label])
    print("agriculture_land_count:", agriculture_land_count, 'total:', len(all_labels))

    # save the labels to a file
    if labels_present:
        with open(save_path + '/labels.txt', 'w') as f:
            for labels in all_labels:
                labels_str = ','.join(labels)
                f.write(str(labels_str) + '\n')


def test_lmdb(path='data/deepglobe_patches/train'):
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

    subsample_whole_dir(dir_path="data/deepglobe/valid")
    # test_lmdb()
    # get_labels_from_deepglobe()
