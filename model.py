import numpy as np
from block import *
import utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, stop, n_block=4):
        super(Generator, self).__init__()
        self.stop = stop
       
        if stop == 'up':
            act = 'prelu'
        else:
            act = 'mprelu' 

        self.input_conv = ConvBlock(num_channels, base_filter, 4, 2, 1, activation=act, norm=None, bias=True)
        self.conv1 = ConvBlock(1*base_filter, 2*base_filter, 4, 2, 1, activation=act, norm='batch', bias=False) 
        self.conv2 = ConvBlock(2*base_filter, 4*base_filter, 4, 2, 1, activation=act, norm='batch', bias=False)
        self.conv3 = ConvBlock(4*base_filter, 8*base_filter, 4, 2, 1, activation=act, norm='batch', bias=False) 
        self.conv4 = ConvBlock(8*base_filter, 8*base_filter, 4, 2, 1, activation=act, norm='batch', bias=False) 

        self.deconv4 = Upsample2xBlock(8*base_filter, 8*base_filter, activation=act, norm='batch', bias=False, upsample='rnc')
        self.deconv5 = Upsample2xBlock(16*base_filter, 4*base_filter, activation=act, norm='batch', bias=False, upsample='rnc')
        self.deconv6 = Upsample2xBlock(8*base_filter,2*base_filter, activation=act, norm='batch', bias=False, upsample='rnc')
        self.deconv7 = Upsample2xBlock(4*base_filter, 1*base_filter, activation=act, norm='batch', bias=False, upsample='rnc')
        self.output_deconv = Upsample2xBlock(2*base_filter, num_channels, activation=act, norm='batch', bias=False, upsample='rnc')  
        self.output = ConvBlock(2*num_channels, num_channels, 3, 1, 1, activation='tanh', norm=None, bias=False) 
 
    def forward(self, x):
        e1 = self.input_conv(x)
        e2 = self.conv1(e1)
        e3 = self.conv2(e2)
        e4 = self.conv3(e3)
        e5 = self.conv4(e4)

        d4 = F.dropout(self.deconv4(e5), 0.5, training=True)
        d4 = torch.cat([d4, e4], 1)

        d5 = F.dropout(self.deconv5(d4), 0.5, training=True)
        d5 = torch.cat([d5, e3], 1)

        d6 = F.dropout(self.deconv6(d5), 0.5, training=True)
        d6 = torch.cat([d6, e2], 1)

        d7 = self.deconv7(d6)
        d7 = torch.cat([d7, e1], 1)

        d8 = self.output_deconv(d7)

        in_out = torch.cat([d8, x], 1)
        out = self.output(in_out)
        return out      

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

# Defines the PatchGAN discriminator.
class NLayerDiscriminator(nn.Module):
    def __init__(self, num_channels, base_filter, image_size, n_layers=4):
        super(NLayerDiscriminator, self).__init__()
      
        kw = 4
        padw = 1

        # global feature extraction
        sequence = [
            nn.Conv2d(num_channels, base_filter, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(base_filter * nf_mult_prev, base_filter * nf_mult, kernel_size=kw, stride=2,
                          padding=padw), nn.BatchNorm2d(base_filter * nf_mult,
                                                    affine=True), nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(base_filter * nf_mult_prev, base_filter * nf_mult, kernel_size=kw, stride=1,
                      padding=padw), nn.BatchNorm2d(base_filter * nf_mult,
                                                affine=True), nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(base_filter * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=0.9, target_fake_label=0.1,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
