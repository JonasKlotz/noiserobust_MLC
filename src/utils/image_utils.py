import os.path

import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def show(imgs, index=0):
    """
    Plots images from Pytorch Tensor format to matplotlib
    :param imgs:
    :return:
    """
    n_img = imgs.shape[0]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i in range (n_img):
        img= imgs[i,...]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
        if torch.min(img) < 0:
            img = ((img+1)*128).to(torch.int)

        axs[0, i].imshow(np.asarray(img), vmin=0, vmax=255)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(f"results/img_{index}.png")
    plt.show()


def plot_img_and_bars(img, prediction, results, labels, loss, index=0):

    fig, axs = plt.subplots(ncols=2, squeeze=False, figsize=(20,10))

    ############# img part #################
    img = img[0, ...]
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0) * imagenet_stats[1][0] + imagenet_stats[0][0]
        img = torch.clip(img, 0., 1.)

    axs[0, 0].imshow(np.asarray(img), vmin=0, vmax=1)
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    ############### barplot ###################
    prediction = (prediction.flatten().numpy()).copy()
    results = (results.flatten().numpy()).copy()
    # Using numpy to group 3 different data with bars
    X = np.arange(len(labels)) + 1
    # Passing the parameters to the bar function, this is the main function which creates the bar plot

    # Using X now to align the bars side by side
    axs[0, 1].bar(X, prediction, color='lightblue', width=0.3)
    axs[0, 1].bar(X + 0.3, results, color='grey', width=0.3)

    # Creating the legend of the bars in the plot
    axs[0, 1].legend(['Prediction', 'True Label'])

    # Overiding the x axis with the country names
    axs[0, 1].set_xticks(ticks=X, labels=labels)
    # Giving the tilte for the plot
    axs[0, 1].set_title(f"Barplot with total BCE Loss of {loss}")
    # Namimg the x and y axis
    axs[0, 1]. set_xlabel('Labels')
    axs[0, 1]. set_ylabel('Predicted Accuracy')
    # Saving the plot as a 'png'
    fig.savefig(f'plots/boxplots/boxplot_{index}.png')
    # Displaying the bar plot
    plt.show()


def barplot_results(prediction, results, labels, loss, index=0):
    # Declaring the figure or the plot (y, x) or (width, height)
    fig = plt.figure(figsize=[15, 10])

    # Data to be plotted
    prediction = (prediction.flatten().numpy()).copy()
    results = (results.flatten().numpy()).copy()
    # Using numpy to group 3 different data with bars
    X = np.arange(len(labels))+1
    # Passing the parameters to the bar function, this is the main function which creates the bar plot

    # Using X now to align the bars side by side
    plt.bar(X, prediction, color='lightblue', width=0.3)
    plt.bar(X + 0.3, results, color='grey', width=0.3)

    # Creating the legend of the bars in the plot
    plt.legend(['Prediction', 'True Label'])

    # Overiding the x axis with the country names
    plt.xticks(ticks=X, labels=labels)
    # Giving the tilte for the plot
    plt.title(f"Barplot with total BCE Loss of {loss}")
    # Namimg the x and y axis
    plt.xlabel('Labels')
    plt.ylabel('Predicted Accuracy')
    # Saving the plot as a 'png'
    fig.savefig(f'results/boxplot_{index}.png')
    # Displaying the bar plot
    return fig


def plot_confusion_matrix(confusion_matrix, axes, class_label, class_names, epoch=0, fontsize=14):
    """
    plot a confusion matrix

    :param confusion_matrix:
    :param axes:
    :param class_label: title for plot
    :param class_names: x and y ticks for plot
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, dtype=int
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for class - " + class_label + " in epoch " + str(epoch))


def plot_multilabel_confusion_matrices(conf_mats, class_names, dir_name="", epoch=0, show=False):
    """

    :param conf_mats: NP array containing multilabel confusion matrices
    :param class_names: labels
    :param title: model name
    :param epoch:
    """
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    ax = ax.flatten()
    for i in range(len(class_names)):
        plot_confusion_matrix(conf_mats[i], ax[i], class_names[i], ["N", "Y"], epoch)

    for i in range(len(class_names), 4 * 5):
        ax[i].axis('off')
    fig.tight_layout()
    fig.savefig(dir_name + f"/CM_epoch_{epoch}.png")
    if show:
        plt.show()


def save_confusion_matrix(y_true, y_pred, class_names, epoch=1, dir_name="", every_nth_epoch=0):
    """

    Calculate and save confusion Matrix


    :param y_true:
    :param y_pred:
    :param class_names:
    :param epoch:
    :param model_name:
    """
    # save confusion matrix every nth epoch only, otherwise files get too big
    if every_nth_epoch and not epoch % every_nth_epoch == 0:
        return

    try:
        dir_name = os.path.join(dir_name, "conf_mat")
        os.makedirs(dir_name)
    except OSError as exc:
        pass

    conf_mats = multilabel_confusion_matrix(y_true, y_pred)
    plot_multilabel_confusion_matrices(conf_mats, class_names, dir_name=dir_name, epoch=epoch, show=False)
