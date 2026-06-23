import torch
import torch.nn as nn
import modules.convolution as cc


class AlexNetCIFAR10(nn.Module):
    def __init__(
        self,
        multiplier_matrix,
        num_classes: int = 10,
        conv_type: int = 1,
        bit_width: int = 8,
        signed: bool = False,
        zone: bool = False,
        dropout: float = 0.5,
        classifier_dim: int = 1024,
    ):
        super().__init__()

        self.conv1 = cc.Conv2d_custom(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name="s", multiplier_matrix=multiplier_matrix,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = cc.Conv2d_custom(
            64, 128, kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name="2", multiplier_matrix=multiplier_matrix,
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = cc.Conv2d_custom(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name="3", multiplier_matrix=multiplier_matrix,
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = cc.Conv2d_custom(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name="4", multiplier_matrix=multiplier_matrix,
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=False)

        self.conv5 = cc.Conv2d_custom(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name="5", multiplier_matrix=multiplier_matrix,
        )
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=False)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        last_feat_dim = 256 * 4 * 4
        self.drop1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(last_feat_dim, classifier_dim)
        self.relu_fc1 = nn.ReLU(inplace=False)
        self.drop2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(classifier_dim, classifier_dim)
        self.relu_fc2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(classifier_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        return x