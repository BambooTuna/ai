import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 9), padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(5, 5), padding=2)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, val=0)
