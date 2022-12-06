'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from hsic import hsic_normalized_cca


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []
        
        out = F.relu(self.bn1(self.conv1(x)))
        # output_list.append(out)
        
        out = self.bn2(self.conv2(out))
       #  output_list.append(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        output_list.append(out)
        
        return out, output_list

    # only conv in block
    # def forward(self, x):
    #     if isinstance(x, tuple):
    #         x, output_list = x
    #     else:
    #         output_list = []
        
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     output_list.append(out)
        
    #     out = self.bn2(self.conv2(out))
    #     output_list.append(out)
        
    #     out += self.shortcut(x)
    #     out = F.relu(out)
    #     # output_list.append(out)
        
    #     return out, output_list

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        output_list.append(out)

        return out, output_list


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        expansion = block.expansion

        self.avgpool = torch.nn.AdaptiveAvgPool2d((4,4))

        self.linear = nn.Linear(512*expansion, args.n_classes)
        
        self.record = False
        self.targets = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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


    def fc_filter(self, X, cov_fea):
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

        num_filtered = 20

        for i in range(num_filtered):
            idy = y_list[i][0]
            mask[:,idy:idy+1] *= 0

            idx = x_list[len(x_list)-1-i][0]
            mask[:,idx:idx+1] *= 0

        return mask.cuda()


    def forward(self, x):
        output_list = []
        
        out = F.relu(self.bn1(self.conv1(x)))
        output_list.append(out)
        
        out, out_list = self.layer1(out)
        output_list.extend(out_list)
        
        out, out_list = self.layer2(out)
        output_list.extend(out_list)
        
        out, out_list = self.layer3(out)
        output_list.extend(out_list)
        
        out, out_list = self.layer4(out)
        output_list.extend(out_list)

        out = self.avgpool(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        output_list.append(out)

        if self.targets is not None:
            mask = self.fc_filter(x, out)
            out = out * mask
            self.targets = None

        out = F.log_softmax(self.linear(out), dim=1)
            
        if self.record:
            self.record = False
            return out, output_list
        else:
            return out


def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args)

def ResNet34(args):
    return ResNet(BasicBlock, [3,4,6,3], args)

def ResNet50(args):
    return ResNet(Bottleneck, [3,4,6,3], args)

def ResNet101(args):
    return ResNet(Bottleneck, [3,4,23,3], args)

def ResNet152(args):
    return ResNet(Bottleneck, [3,8,36,3], args)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

# model = ResNet50()
# for name, w in model.named_parameters():
#     if(len(w.size()) == 4 and "shortcut" not in name):
#         print(name, w.size())
