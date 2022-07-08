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

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(num_ftrs, d_model) # out features are model dim


    def forward(self, img):
        x = self.model(img)
        print(x.shape)
        output = x.view(img.size(0), 1, -1) # why this transformation??
<<<<<<< HEAD

=======
        _output = output.detach().numpy().reshape(-1, output.size(2))
        _img = img.detach().numpy()[:,0,:,:].reshape(-1, img.size(2)*img.size(3))
>>>>>>> b1a3e7a522506fe8d91d036539ba0d077b661690
        return output
