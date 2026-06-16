import torch
import torch.nn as nn
import modules.convolution as cc


class VGG16(nn.Module):
    def __init__(self, multiplier_matrix, num_classes=10, conv_type = 1, bit_width = 8, signed = False, zone = False):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            cc.Conv2d_custom(3, 64, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "1", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            cc.Conv2d_custom(64, 64, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "2", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            cc.Conv2d_custom(64, 128, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "3", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            cc.Conv2d_custom(128, 128, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "4", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            cc.Conv2d_custom(128, 256, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "5", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            cc.Conv2d_custom(256, 256, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "6", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            cc.Conv2d_custom(256, 256, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "7", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            cc.Conv2d_custom(256, 512, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "8", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            cc.Conv2d_custom(512, 512, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "9", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            cc.Conv2d_custom(512, 512, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "10", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            cc.Conv2d_custom(512, 512, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "11", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            cc.Conv2d_custom(512, 512, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "12", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            cc.Conv2d_custom(512, 512, kernel_size=3, stride=1, padding=1, bias=True,
                conv_type=conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "13", multiplier_matrix=multiplier_matrix),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out