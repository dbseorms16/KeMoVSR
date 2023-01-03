import logging
from collections import OrderedDict
from re import X
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import numpy as np

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()

        self.ResBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, padding=3),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.ResBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=5, padding=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.ResBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(in_channels=3*channel, out_channels=channel, kernel_size=3, padding=1) 
        
    def forward(self, x):
        
        residual = x
        x1 = self.ResBlock1(x)
        x1 += residual
        
        x2 = self.ResBlock2(x)
        x2 += residual

        x3 = self.ResBlock3(x)
        x3 += residual
        
        out = torch.cat([x1,x2,x3], axis=1)
        out = self.conv(out)

        return out

class Org_ResBlock(nn.Module):
    def __init__(self, channel):
        super(Org_ResBlock, self).__init__()

        self.ResBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1) 
        
    def forward(self, x):
        
        residual = x
        x1 = self.ResBlock1(x)
        x1 += residual
        
        out = self.conv(x1)

        return out

class SplitModule(nn.Module):
    def __init__(self, channel, split_num):
        super(SplitModule, self).__init__()

        self.channel = channel
        self.split_num = split_num
        splits = [1 / split_num] * split_num
        self.share = int(self.channel / self.split_num)
        self.mod = int(self.channel % self.split_num)
        self.in_split = []


        self.Match = nn.Conv2d(in_channels=self.channel, out_channels=(self.channel-self.mod), kernel_size=3, padding=1)
        
        self.AffineModule = nn.Sequential(
            nn.Conv2d(in_channels=(int(self.channel/self.split_num)), out_channels=(int(self.channel/self.split_num)), kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=(int(self.channel/self.split_num)), out_channels=(int(self.channel/self.split_num)), kernel_size=1),
        )

        for i in range(self.split_num):
            in_split = round(channel * splits[i]) if i < self.split_num - 1 else channel - sum(self.in_split)
            self.in_split.append(in_split)
            setattr(self, 'fc{}'.format(i),nn.Sequential(*[
            nn.Conv2d(in_channels=(int(self.channel/self.split_num)), out_channels=(int(self.channel/self.split_num)), kernel_size=1),
            # nn.BatchNorm2d(int(self.channel/self.split_num)),
            nn.ReLU(),
            nn.Conv2d(in_channels=(int(self.channel/self.split_num)), out_channels=(int(self.channel/self.split_num))*2, kernel_size=1),
            ]))

            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=int(self.channel/self.split_num), out_channels=int(self.channel/self.split_num), kernel_size=3, padding=1))
        


    def forward(self, x):
        input = torch.split(x, self.in_split, dim=1)

        out = []
        for i in range(self.split_num):
            s, t = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            out.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(s) + t))

        return torch.cat(out, 1)


class SPBlock(nn.Module):
    ''' Residual block based on  '''
    def __init__(self, channel, split_num=2):
        super(SPBlock, self).__init__()

        self.res = nn.Sequential(*[
            SplitModule(channel = channel, split_num = split_num),
            nn.ReLU(inplace=True),
            SplitModule(channel = channel, split_num = split_num),
        ])

    def forward(self, x):
        return x + self.res(x)




class KernelEstimation(nn.Module):
    ''' Network of non_parameteric_regression'''
    def __init__(self, in_nc=3, kernel_size=21, channels=[128, 256, 128, 64, 32], split_num=2):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.head = nn.Conv2d(in_channels=in_nc, out_channels=channels[0], kernel_size=3, padding=1, bias=True)
        
        self.RB1 = ResBlock(channel=channels[0])
        self.SP1 = SPBlock(channel=channels[0], split_num=2)
        
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, bias=True)
        self.RB2 = ResBlock(channel=channels[1])
        self.SP2 = SPBlock(channel=channels[1], split_num=2)
        
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, padding=1, bias=True)
        self.RB3 = ResBlock(channel=256)
        self.SP3 = SPBlock(channel=256, split_num=2)

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=channels[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[3], out_channels=441, kernel_size=3, padding=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.softmax = nn.Softmax(1)
        self.linear1 = nn.Linear(128, 3)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.PReLU()


        self.up = sequential(nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0],
                                                   kernel_size=2, stride=2, padding=0, bias=True),
                                *[ResBlock(channel = channels[0])])

        self.down = sequential(*[ResBlock(channel = channels[0])],
                                  nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=2, stride=2, padding=0,
                                            bias=True))
        
        self.scale = 4
        self.kernel_size = 21



    def forward(self, x):
        print(x.size())
        b, c, h, w = x.size()
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
       
        x = self.head(x)
        x = self.RB1(x)
       
        multiple = self.SP1(x)
        x = x + multiple
        x = self.conv1(x)
        x = self.RB2(x)
        
        multiple2 = self.SP2(x)
        x = x + multiple2
        x = self.conv2(x)
        x = self.RB3(x)
        
        multiple3 = self.SP3(x)
        x = x + multiple3
        x = self.tail(x)
        x = x[..., :h, :w]

        x = self.softmax(x)
        x = self.avg_pool(x)
        x = x.view(b, -1, self.kernel_size, self.kernel_size)

        return x