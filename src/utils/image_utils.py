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
        if torch.min(img) < 0:
            img = ((img+1)*128).to(torch.int)

        axs[0, i].imshow(np.asarray(img), vmin=0, vmax=255)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()