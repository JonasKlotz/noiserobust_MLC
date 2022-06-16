import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np




def show(imgs):
    """
    Plots images from Pytorch Tensor format to matplotlib
    :param imgs:
    :return:
    """
    n_img = imgs.shape[0]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i in range (n_img):
        img= imgs[i,...]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()