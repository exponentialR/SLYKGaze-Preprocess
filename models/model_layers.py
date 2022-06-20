import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
Pool = nn.MaxPool2d

class NetMix():
    """
    Utility to load a weight file to a device.
    """
    def load_weight(self, weight_pt, device):
        state_dict = torch.load(weight_pt, map_location = device )
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        weights = {}
        for keys in state_dict:
            m = re.search (r'(^fc\.|\.fc\.|^features\.|\.features\.)', keys)
            if m is None: continue
            new_key = keys[m.start():]
            new_key = new_key[1:] if new_key[0] == '.' else new_key
            weights[new_key] = state_dict[k]

        #load weights and set model to eval()
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using image embedding network pretrained weight: {Path(weight_pt).name}')
        return self

    def set_trainable(self, trainable = False):
        for param in self.parameters():
            paran.requires_grad = trainable

# class ConvGaze(nn.Module, NetMix):
class ConvGaze (nn.Module, NetMix):

    def __init__(self, input_dim, output_dim, kernel_size = 3, stride = 1, batch_norm = False, relu =True):
        super(ConvGaze, self).__init__()
        self.input_dimension = input_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding = (kernel_size -1)//2, bias = True)
        self.relu = None
        self.batch_norm = None
        if relu:
            self.relu = nn.ReLU()
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        assert x.size()[1] == self.input_dimension, '{} {}'.format(x.size()[1], self.input_dimension)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = ConvGaze(input_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = ConvGaze(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim /2))
        self.conv3 = ConvGaze(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = ConvGaze(input_dim, out_dim, 1, relu=False)
        if input_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        output = self.bn1(x)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn3(output)
        output = self.relu(output)
        output = self.conv3(output)
        output += residual
        return output

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase = 0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)

        #lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n

        #recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn = bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = nn.functional.interpolate(low3, x.shape[2:], mode ='bilinear')
        return up1 + up2

