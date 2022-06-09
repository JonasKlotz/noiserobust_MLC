from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torchvision.models as models


class ResnetEncoder(nn.Module):
    """
    Pytorch module for a resnet encoder

    """

    def __init__(
            self, config:dict):
        super(ResnetEncoder, self).__init__()

        model = models.resnet101(pretrained=config["pretrained"])
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(2048, config["output_dimension"])

    def forward(self, input_image):
        x = self.features(input_image)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.view(input_image.size(0), 1, -1), None
