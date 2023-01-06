import torch
from torch import nn
from torch.utils.cpp_extension import load

conv2d_cuda = load(name="conv2d_cuda", sources=["conv2d.cpp", "conv2d_kernel.cu"])

# Define ResBlock
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # Residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels * ResBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*ResBlock.expansion)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    # Forward path
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y) 
        y = self.bn2(y)
        y += self.shortcut(x)
        y = self.relu(y)

        return y

# Define ResNet
class ResNet_baseline(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_baseline, self).__init__()
        self.in_channels = 64

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    # Helper function
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block.expansion
        return nn.Sequential(*layers)

    # Forward Path
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    # Helper function
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block.expansion
        return nn.Sequential(*layers)

    # Forward Path
    def forward(self, x):
        y = conv2d_cuda.forward(x, self.conv1.weight, self.conv1.padding[0], self.conv1.stride[0])
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

def ResNet18_baseline(num_classes):
    return ResNet_baseline(ResBlock, [2,2,2,2], num_classes)

def ResNet18(num_classes):
    return ResNet(ResBlock, [2,2,2,2], num_classes)