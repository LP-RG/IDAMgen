import torch
import torch.nn as nn
import torch.nn as nn
import modules.convolution as cc


class LeNet5(nn.Module):
    def __init__(self, multiplier_matrix, num_classes=10, conv_type = 1, bit_width = 8, signed = False, zone = False):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
        cc.Conv2d_custom(1, 6, kernel_size=5, stride=1, padding=0, bias=True,
                            conv_type=conv_type, bit_width=bit_width, signed= signed, name = "1", multiplier_matrix = multiplier_matrix),
        nn.BatchNorm2d(6),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
        cc.Conv2d_custom(6, 16, kernel_size=5, stride=1, padding=0, bias=True,
                        conv_type=conv_type, bit_width=bit_width, signed= signed, name = "1", multiplier_matrix = multiplier_matrix),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out