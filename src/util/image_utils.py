import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np




def show(imgs):
    """
    Plots images from Pytorch Tensor format to matplotlib
    :param imgs:
    :return:
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])