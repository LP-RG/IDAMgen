import torch
import torch.nn as nn
import modules.convolution as cc

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, multiplier_matrix, stride=1,
                 conv_type=1, bit_width=8, signed=False, name=""):
        super().__init__()
        self.conv1 = cc.Conv2d_custom(in_ch, out_ch, 3, stride=stride, padding=1, bias=False,
                                      conv_type=conv_type, bit_width=bit_width, signed=signed,
                                      name=name+"_3x3a", multiplier_matrix=multiplier_matrix)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = cc.Conv2d_custom(out_ch, out_ch, 3, stride=1, padding=1, bias=False,
                                      conv_type=conv_type, bit_width=bit_width, signed=signed,
                                      name=name+"_3x3b", multiplier_matrix=multiplier_matrix)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                cc.Conv2d_custom(in_ch, out_ch, 1, stride=stride, padding=0, bias=False,
                                 conv_type=conv_type, bit_width=bit_width, signed=signed,
                                 name=name+"_sc", multiplier_matrix=multiplier_matrix),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.shortcut(x) if not isinstance(self.shortcut, nn.Identity) else x)
        return self.relu(out)

class ResNet56_CIFAR100(nn.Module):
    def __init__(
        self,
        multiplier_matrix,
        num_classes=10,
        conv_type=1,
        bit_width=8,
        signed=False,
        zone=False,
    ):
        super().__init__()
        widths = [16, 32, 64]
        self.conv1 = cc.Conv2d_custom(3, widths[0], 3, stride=1, padding=1, bias=False,
                                      conv_type=conv_type, bit_width=bit_width, signed=signed,
                                      name="s", multiplier_matrix=multiplier_matrix)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(widths[0], widths[0], blocks=9, stride=1, base="2",
                                       multiplier_matrix=multiplier_matrix, conv_type=conv_type, bit_width=bit_width, signed=signed)
        self.layer2 = self._make_layer(widths[0], widths[1], blocks=9, stride=2, base="3",
                                       multiplier_matrix=multiplier_matrix, conv_type=conv_type, bit_width=bit_width, signed=signed)
        self.layer3 = self._make_layer(widths[1], widths[2], blocks=9, stride=2, base="4",
                                       multiplier_matrix=multiplier_matrix, conv_type=conv_type, bit_width=bit_width, signed=signed)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, in_ch, out_ch, blocks, stride, base,
                    multiplier_matrix, conv_type, bit_width, signed):
        layers = [BasicBlock(in_ch, out_ch, multiplier_matrix, stride,
                             conv_type, bit_width, signed, name=f"{base}a")]
        for i in range(1, blocks):
            suffix = chr(ord('a') + i)
            layers.append(BasicBlock(out_ch, out_ch, multiplier_matrix, 1,
                                     conv_type, bit_width, signed, name=f"{base}{suffix}"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avg_pool(x); x = torch.flatten(x, 1)
        return self.fc(x)
