import json
import os
import numpy as np
from matplotlib import pyplot as plt

from data_pipeline.deepglobe.patch_sampling import LABELS


def get_labels_from_deepglobe(path='data/deepglobe/train'):
    """ Read all mask images and get the pixels """

    images = [f for f in os.listdir(path) if f.endswith('mask.png')]
    for image_file in images:

        label_img = plt.imread(f"{path}/{image_file}")

        # reshape the label image to a 2D array of pixels
        label_pixels = label_img.reshape(-1, 3).tolist()

        unique_pixels = np.unique(label_pixels, axis=0)

        # only keep unique labels from the list
        label_pixels_str = [str(pixel) for pixel in unique_pixels]
        labels = list(set(label_pixels_str))

        # map the labels to their names
        label_names = [LABELS[str(label)] for label in labels]
        # append file with labels
        with open(f"results/data/deepglobe_labels.txt", 'a') as f:
            labels_str = ','.join(label_names)
            f.write(f"{labels_str}\n")
            print(labels_str)


def get_class_distribution(txt_path='results/data/deepglobe-patches_labels.txt'):
    """ Plot the distribution of labels in the dataset """

    # read the labels from the file
    with open(txt_path, 'r') as f:
        # get the number of lines
        labels = f.read().split('\n')
        num_samples = len(labels)

    single_labels = [label.split(',') for label in labels]

    # count each label in the file
    label_stats = {'count': {}, 'percentage': {}}
    for label in LABELS.values():
        count = sum([label in label_list for label_list in single_labels])
        label_stats['count'][label] = count
        label_stats['percentage'][label] = count / num_samples
        print(f"{label}: {count}")

    # sum all counts
    label_stats['total_count'] = sum(label_stats['count'].values())
    label_stats['avg_classes_present'] = sum(label_stats['percentage'].values())
    print(f"Total count: {label_stats['total_count']} Avg classes present: {label_stats['avg_classes_present']}")

    # save the counts to json
    with open(txt_path.replace("labels.txt", 'stats.json'), 'w') as f:
        json.dump(label_stats, f)

    # plot the distribution of percentages
    plt.figure(figsize=(7, 7))
    plt.title("Presence of label classes in dataset", fontsize=18)
    plt.bar(range(len(label_stats['percentage'])), list(label_stats['percentage'].values()), align='center')
    plt.xticks(range(len(label_stats['percentage'])), list(label_stats['percentage'].keys()))
    plt.xlabel("Label Class", fontsize=14)
    plt.ylim(0, 1)
    plt.ylabel("Percentage of Images present", fontsize=14)
    # set x-ticks to be rotated and centered
    plt.xticks(rotation=90, ha='center')
    # make enough space for the x-ticks
    plt.subplots_adjust(bottom=0.22)
    plt.show()


def plot_pixel_legend():
    """ Plot the legend for the pixel colors """

    # create a figure with a single subplot
    plt.figure(figsize=(7, 7))
    plt.title("Pixel Legend", fontsize=18)

    # create a subplot with with rgb values for each label
    for label in LABELS.keys():
        rgb = label.replace('[', '').replace(']', '').split('. ')
        # convert to ints
        rgb = [float(rgb_val) for rgb_val in rgb]
        # plot square of size 1 with the rgb values
        plt.plot([0, 1], [0, 1], color=rgb, linewidth=1000)
        plt.title(f"{rgb}")
        plt.show()




if __name__ == '__main__':
    # get_labels_from_deepglobe()

    # get_class_distribution(txt_path='results/data/deepglobe_labels.txt')
    # get_class_distribution(txt_path='results/data/deepglobe-patches_labels.txt')
    plot_pixel_legend()