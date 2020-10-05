import torch
import numpy as np


class MPReLU(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(MPReLU, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        return -torch.nn.functional.prelu(-input, self.weight)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + 'num_parameters=' + str(self.num_parameters) + ')'


class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='prelu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'mprelu':
            self.act = MPReLU() 

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'mprelu':
            self.act = MPReLU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='rnc', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        
        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            ConvBlock(input_size, output_size,
                      kernel_size=3, stride=1, padding=1,
                      bias=bias, activation=activation, norm=norm)
        )

    def forward(self, x):
        out = self.upsample(x)
        return out


