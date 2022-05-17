import decoder
from encoder import ResnetEncoder
import torch.nn as nn


class MultiLabelClassifier(nn.Module):
    """

    """

    def __init__(self, **kwargs):
        super().__init__()
        resnet_layers = 101
        self.encoder = ResnetEncoder(num_layers=resnet_layers, pretrained=False)
        self.decoder = None

    def forward(self, input_image):
        pass
