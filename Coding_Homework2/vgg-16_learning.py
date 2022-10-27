import torch
import torch.nn as nn

# The structure of VGG-X, containing the out_channels and 'M'
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# The functions to make the layers:
    def _make_layers(self, layer):
        #define a functions list
        layers = []
        in_channels = 3
        # For each element of cfg, we do
        for x in layer:
            if x == 'M':
                #If 'M', we add maxPooling in layers
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                #If x is out_channels number, we build this block, which is conv2d + batch norm + ReLu
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                #Then, set next in_channels to be out_channels
                in_channels = x
        # At last, add avgPooling
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16')
    x = torch.randn(2,3,32,32)
    y = net(x)


