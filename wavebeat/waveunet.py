import os
import torch
import argparse
import numpy as np
import pytorch_lightning as pl

from wavebeat.base import Base
from wavebeat.utils import center_crop, causal_crop

class DSBlock(torch.nn.Module):
    def __init__(self, 
                ch_in, 
                ch_out, 
                kernel_size=15, 
                stride=2):
        super(DSBlock, self).__init__()

        assert(kernel_size % 2 != 0)    # kernel must be odd length
        padding = kernel_size//2        # calculate same padding

        self.conv1 = torch.nn.Conv1d(ch_in, 
                                     ch_out, 
                                     kernel_size=kernel_size, 
                                     padding=padding)
        self.bn    = torch.nn.BatchNorm1d(ch_out)
        self.prelu = torch.nn.PReLU(ch_out)
        self.conv2 = torch.nn.Conv1d(ch_out, ch_out, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.prelu(x)
        x_ds = self.conv2(x)
        return x_ds, x

class USBlock(torch.nn.Module):
    def __init__(self, 
                ch_in, 
                ch_out, 
                kernel_size=5, 
                scale_factor=2, 
                skip="add"):
        super(USBlock, self).__init__()

        assert(kernel_size % 2 != 0)    # kernel must be odd length
        padding = kernel_size//2        # calculate same padding

        self.skip = skip
        self.conv  = torch.nn.Conv1d(ch_in, 
                                     ch_out, 
                                     kernel_size=kernel_size, 
                                     padding=padding)
        self.bn    = torch.nn.BatchNorm1d(ch_out)
        self.prelu = torch.nn.PReLU(ch_out)
        self.us    = torch.nn.Upsample(scale_factor=scale_factor)

    def forward(self, x, skip):
        x = self.us(x) # upsample by x2

        # handle skip connections
        if   self.skip == "add":    x = x + skip
        elif self.skip == "concat": x = torch.cat((x,skip), dim=1)
        elif self.skip == "none":   pass
        else:                       raise NotImplementedError()

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x

class WaveUNetModel(Base):
    """ Wave-U-Net with linear upsampling.

    Args:
        ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
        noutputs (int): Number of output channels, beats and downbeats. Default: 2
        nblocks (int): Number of total DS/US blocks. Default: 12
        ds_kernel (int): Width of the downsampling convolutional kernels. Default: 15
        us_kernel (int): Width of the upsampling convolutional kernels. Default: 5
        out_kernel (int): Width of the output convolutional kernel. Default: 3
        channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
        causal (bool): Causal configuration does not consider future input values. Default: False
        skip (str): Skip connection type from each DS block to the US block 'add', 'concat', 'none'. Default: 'add'
        stride (int): Downsampling (and upsampling) factor. Default: 2
    """
    def __init__(self, 
                ninputs = 1,
                noutputs = 2,
                nblocks = 12,
                ds_kernel = 15,
                us_kernel = 5,
                out_kernel = 3,
                channel_growth = 24,
                skip = "add",
                stride = 2,
                causal = False, 
                **kwargs):
        super(WaveUNetModel, self).__init__()
        self.save_hyperparameters()

        self.encoder = torch.nn.ModuleList()
        for n in np.arange(self.hparams.nblocks):
            ch_in = n * self.hparams.channel_growth
            ch_out = (n+1) * self.hparams.channel_growth
            if ch_in == 0: ch_in = 1 # mono input
            self.encoder.append(DSBlock(ch_in, ch_out,
                                kernel_size=self.hparams.ds_kernel))

        self.embedding = torch.nn.Conv1d(ch_out, ch_out, kernel_size=1)

        self.decoder = torch.nn.ModuleList()
        for n in np.arange(self.hparams.nblocks, stop=0, step=-1):
            ch_in = n * self.hparams.channel_growth
            ch_out = (n-1) * self.hparams.channel_growth
            if ch_out == 0: ch_out = self.hparams.channel_growth
            if self.hparams.skip == "concat": ch_in *= 2
            self.decoder.append(USBlock(ch_in, ch_out,
                                kernel_size=self.hparams.us_kernel,
                                skip=self.hparams.skip))

        self.output_conv = torch.nn.Conv1d(ch_out, 2, 
                            kernel_size=self.hparams.out_kernel)

        #self.pool = torch.nn.MaxPool1d(1023, stride=220, padding=1023//2)

    def forward(self, x):

        x_in = x
        skips = []

        for enc in self.encoder:
            x, skip = enc(x)
            skips.append(skip)

        x = self.embedding(x)

        for dec in self.decoder:
            skip = skips.pop()
            x = dec(x, skip)

        #x = self.pool(x)
        x = self.output_conv(x)
        x = torch.sigmoid(x)

        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--nblocks', type=int, default=12)
        parser.add_argument('--channel_growth', type=int, default=24)
        parser.add_argument('--skip', type=str, default="add")
        parser.add_argument('--stride', type=int, default=2)
        parser.add_argument('--ds_kernel', type=int, default=15)
        parser.add_argument('--us_kernel', type=int, default=5)
        parser.add_argument('--out_kernel', type=int, default=3)

        return parser
