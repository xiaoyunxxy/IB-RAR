import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from hsic import hsic_normalized_cca

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []

        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        output_list.append(out)
        return out, output_list

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((8,8))
        self.linear = nn.Linear(nStages[3], num_classes)
        self.record = False
        self.targets = None

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def fc_filter(self, X, cov_fea, rb=True, num_filtered=20):
        mask = torch.ones(cov_fea.shape)

        mi_list = []
        x = X.view(X.shape[0], -1)
        y = self.targets

        for i in range(cov_fea.shape[1]-1):
            fc_i = cov_fea[:,i:i+1].view(cov_fea.shape[0], -1)
            mi_xt = hsic_normalized_cca(x, fc_i, sigma=5)
            mi_yt = hsic_normalized_cca(y.float(), fc_i, sigma=5)
            mi_list.append((i, mi_xt, mi_yt))

        x_list = sorted(mi_list, key=lambda x:x[1])
        y_list = sorted(mi_list, key=lambda x:x[2])

        if rb:
            for i in range(num_filtered):
                idy = y_list[i][0]
                mask[:,idy:idy+1] *= 0

                idx = x_list[len(x_list)-1-i][0]
                mask[:,idx:idx+1] *= 0
        if not rb:
            for i in range(num_filtered):
                idy = y_list[i][0]
                mask[:,idy:idy+1] *= 2

        return mask.cuda()


    def forward(self, x):
        output_list = []

        out = self.conv1(x)
        output_list.append(out)

        out, out_list = self.layer1(out)
        output_list.extend(out_list)

        out, out_list = self.layer2(out)
        output_list.extend(out_list)

        if self.targets is not None:
            mask = self.fc_filter(x, out, rb=True, num_filtered=10)
            out = out * mask
            self.targets = None


        out, out_list = self.layer3(out)
        output_list.extend(out_list)

        out = self.avgpool(out)

        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        output_list.append(out)

        out = self.linear(out)

        if self.record:
            self.record = False
            return out, output_list
        else:
            return out

if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    x = torch.randn(1,3,32,32)
    y = net(x)

    print(y.size())