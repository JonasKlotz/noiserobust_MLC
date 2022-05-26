from matplotlib import pyplot as plt
import os
import numpy as np
import lmdb
import pickle


class Subsampler:
    def __init__(self, img, img_labels, save_dir, patch_size=256, patches=12):
        self.img = img
        self.img_labels = img_labels
        self.save_path = save_dir
        self.size = patch_size
        self.patches = patches

        assert self.img_labels is None or self.img_labels.shape[0] == self.img.shape[0]

    def save_subsamples_to_lmdb(self):
        """
        Saves the subsamples to lmdb files.

        :return:
        """

        subsamples = self._get_subsamples()

        for i, subsample in enumerate(subsamples):
            self._save_single_subsample(subsample, str(i))

    def _save_single_subsample(self, subsample, name, map_size=983298 + 1000):
        """
        Saves the subsamples as lmdb files.

        The map_size is the tested size of one subsample.

        :return:
        """

        # create the save path if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # create the lmdb environment
        env = lmdb.open(os.path.join(self.save_path, name), map_size=map_size)

        # create the lmdb transaction
        with env.begin(write=True) as txn:
            key = name
            value = subsample
            value_dump = pickle.dumps(value)
            # print(len(value_dump))
            txn.put(key.encode('ascii'), )
        env.close()

    def _get_subsamples(self):
        """
        Creates multiple subsamples from a given image. This is achived by a random sampling approach until finished.

        :return: list of subsamples
        """

        subsamples_list = []

        # iterate until the number of patches is reached
        while len(subsamples_list) < self.patches:
            x = int((self.img.shape[0] - self.size) * np.random.random())
            y = int((self.img.shape[1] - self.size) * np.random.random())
            # print(x, y)

            # check if the subsample would overlap with an already existing subsample
            for previous_sub in subsamples_list:
                if previous_sub[0] - self.size <= x <= previous_sub[0] + self.size \
                        and previous_sub[1] - self.size <= y <= previous_sub[1] + self.size:
                    # print("Subsample is too close to another subsample")
                    continue

            # if the subsample is valid, add it to the list
            try:
                subsample = self._get_single_subsample(x, y)
                subsamples_list.append(subsample)
            except Exception as e:
                # print("Invalid subsample", e)
                continue

        return subsamples_list

    def _get_single_subsample(self, x, y):
        """
        Creates a subsample from the class image.

        :param x: x coordinate of the start of the subsample
        :param y: y coordinate of the start of the subsample
        :param size: size of the subsample
        :return: subsample
        """

        assert self.img.shape[0] >= self.size + x, "Invalid x coordinate"
        assert self.img.shape[1] >= self.size + y, "Invalid y coordinate"

        # if labels are given, return their subsample as fourth element
        if self.img_labels is not None:
            return x, y, self.img[x:x + self.size, y:y + self.size], self.img_labels[x:x + self.size, y:y + self.size]
        else:
            return x, y, self.img[x:x + self.size, y:y + self.size],


def subsample_whole_dir(dir_path):

    labels_present = ('train' in dir_path)
    print("Files:", len(os.listdir(dir_path)))

    # iterate over all images in dir that are jpg
    for i, filename in enumerate(os.listdir(dir_path)):
        if filename.endswith('sat.jpg'):

            print(i, filename)

            # get full paths for image loading
            path = dir_path + "/" + filename
            path_labels = path.replace('sat.jpg', 'mask.png')

            save_path = path.replace('deepglobe', 'deepglobe_patches').replace('_sat.jpg', '')

            # load image and labels if exist
            img = plt.imread(path)
            img_labels = plt.imread(path_labels) if labels_present else None

            # create subsampler
            subsampler = Subsampler(img=img, img_labels=img_labels, save_dir=save_path)

            # save subsamples to lmdb
            subsampler.save_subsamples_to_lmdb()


if __name__ == "__main__":
    subsample_whole_dir(dir_path="data/deepglobe/test")
    subsample_whole_dir(dir_path="data/deepglobe/train")
    subsample_whole_dir(dir_path="data/deepglobe/valid")





