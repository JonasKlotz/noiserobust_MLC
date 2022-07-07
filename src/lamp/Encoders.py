import torch.nn as nn
from torchvision import models


class RESNETEncoder(nn.Module):
    def __init__(
            self, d_model=300,  pretrained=True, resnet_layers=18):
        super(RESNETEncoder, self).__init__()
        if resnet_layers == 18:
            self.model = models.resnet18(pretrained=pretrained)
        elif resnet_layers == 50:
            self.model = models.resnet50(pretrained=pretrained)
        else:
            self.model = models.resnet101(pretrained=pretrained)

        # add last layer
        num_ftrs = self.model.fc.in_features # in features
        self.model.fc = nn.Linear(num_ftrs, d_model) # out features are model dim


    def forward(self, img):
        x = self.model(img)
        output = x.view(img.size(0), 1, -1) # why this transformation??
        return output
