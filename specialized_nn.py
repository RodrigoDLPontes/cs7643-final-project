###########################################################################################
## Adapted from the BlazeIt source code (https://github.com/stanford-futuredata/blazeit) ##
###########################################################################################

import math
import torch.nn as nn
import torch.nn.functional as F

class MyModuleList(nn.ModuleList):
    def __add__(self, x):
        tmp = [m for m in self.modules()] + [m for m in x.modules()]
        return MyModuleList(tmp)
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

def make_basic_block(inplanes, planes, stride=1, downsample=None):
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                         stride=stride, padding=1, bias=False)

    block_list = MyModuleList([
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
    ])
    if downsample == None:
        residual = MyModuleList([])
    else:
        residual = downsample
    return MyModuleList([block_list, residual])

# Specialized NN
class PytorchResNet(nn.Module):
    def __init__(self, section_reps=[1,1,1,1],
                 num_classes=7, nbf=16,
                 conv1_size=3, conv1_pad=1,
                 downsample_start=False):
        super(PytorchResNet, self).__init__()

        # Since use_basic_block == True
        self.expansion = 1
        self.block_fn = make_basic_block

        self.downsample_start = downsample_start
        self.inplanes = nbf

        self.conv1 = nn.Conv2d(3, nbf, kernel_size=conv1_size, stride=downsample_start + 1,
                               padding=conv1_pad, bias=False)
        self.bn1 = nn.BatchNorm2d(nbf)

        sections = []
        for i, section_rep in enumerate(section_reps):
            sec = self._make_section(nbf * (2 ** i), section_rep, stride=(i != 0) + 1)
            sections.append(sec)
        self.sections = MyModuleList(sections)
        lin_inp = nbf * int(2 ** (len(section_reps) - 1)) * self.expansion \
            if len(self.sections) != 0 else nbf
        self.fc = nn.Linear(lin_inp, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_section(self, planes, num_blocks, stride=1):
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = MyModuleList([
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
            ])
        else:
            downsample = None

        blocks = []
        blocks.append(self.block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, num_blocks):
            blocks.append(self.block_fn(self.inplanes, planes))

        return MyModuleList(blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.downsample_start:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for sec_ind, section in enumerate(self.sections):
            for block_ind, (block, shortcut) in enumerate(section):
                x_input = x
                if len(shortcut) != 0:
                    x = shortcut(x)
                x_conv = block(x_input)
                x = x + x_conv
                x = F.relu(x)

        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

