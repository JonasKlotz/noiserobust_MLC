import torch.nn as nn
from torchvision import models


class RESNETEncoder(nn.Module):
    def __init__(
            self, d_model=300,  pretrained=True, resnet_layers=18):
        super(RESNETEncoder, self).__init__()
        if resnet_layers == 18:
            model = models.resnet18(pretrained=pretrained)
        elif resnet_layers == 50:
            model = models.resnet50(pretrained=pretrained)
        else:
            model = models.resnet101(pretrained=pretrained)

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
        self.pooling = nn.MaxPool2d(5,5)# 14 14
        self.fc = nn.Linear(2048, d_model)

    def forward(self, img, original_size, return_attns=False):
        #print(f"\ninput shape {img.shape}")
        img = img.view(original_size)
        #print(f"\ntrans formedinput shape {img.shape}")
        x = self.features(img)

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        output = x.view(img.size(0), 1, -1)
        #print(f"output shape {output.shape}")
        return output, None
