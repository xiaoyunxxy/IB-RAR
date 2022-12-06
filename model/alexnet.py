# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
from torch import Tensor
from torch import nn
from hsic import hsic_normalized_cca

__all__ = [
    "AlexNet",
    "alexnet",
]


class AlexNet_cifar(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet_cifar, self).__init__()

        self.record = False
        self.targets = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
            )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )

        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_list = []

        out = self.conv1(x)
        output_list.append(out)

        out = self.conv2(out)
        output_list.append(out)

        out = self.conv3(out)
        output_list.append(out)

        out = self.conv4(out)
        output_list.append(out)

        out = self.conv5(out)

        if self.targets is not None:
            mask = self.fc_filter(x, out, rb=True)
            out = out * mask
            self.targets = None
        output_list.append(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc1(out)
        output_list.append(out)

        out = self.fc2(out)
        output_list.append(out)
        
        out = self.fc3(out)

        if self.record:
            self.record = False
            return out, output_list
        else:
            return out

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()

        self.record = False
        self.targets = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
            )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_list = []

        out = self.conv1(x)
        output_list.append(out)

        out = self.conv2(out)
        output_list.append(out)

        out = self.conv3(out)
        output_list.append(out)

        out = self.conv4(out)
        output_list.append(out)

        out = self.conv5(out)

        if self.targets is not None:
            mask = self.fc_filter(x, out, rb=True)
            out = out * mask
            self.targets = None
        output_list.append(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc1(out)
        output_list.append(out)

        out = self.fc2(out)
        output_list.append(out)

        out = self.fc3(out)

        if self.record:
            self.record = False
            return out, output_list
        else:
            return out


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)

    return model


def alexnet_cifar(**kwargs: Any) -> AlexNet_cifar:
    '''
    modified the first convolutional layer for smaller size of inputs.
    '''
    model = AlexNet_cifar(**kwargs)

    return model