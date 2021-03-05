import torch
from argparse import ArgumentParser

from wavebeat.base import Base

def get_activation(act_type, 
                   ch=None):
    """ Helper function to construct activation functions by a string.

    Args:
        act_type (str): One of 'ReLU', 'PReLU', 'SELU', 'ELU'.
        ch (int, optional): Number of channels to use for PReLU.
    
    Returns:
        torch.nn.Module activation function.
    """

    if act_type == "PReLU":
        return torch.nn.PReLU(ch)
    elif act_type == "ReLU":
        return torch.nn.ReLU()
    elif act_type == "SELU":
        return torch.nn.SELU()
    elif act_type == "ELU":
        return torch.nn.ELU()

class dsTCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size, 
                stride=1,
                dilation=1,
                norm_type=None,
                act_type="PReLU"):
        super(dsTCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type

        pad_value =  ((kernel_size-1) * dilation) // 2

        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     dilation=dilation,
                                     padding=pad_value)
        self.act1 = get_activation(act_type, out_ch)

        if norm_type == "BatchNorm":
            self.norm1 = torch.nn.BatchNorm1d(out_ch)
            #self.norm2 = torch.nn.BatchNorm1d(out_ch)
            self.res_norm = torch.nn.BatchNorm1d(out_ch)
        else:
            self.norm1 = None
            self.res_norm = None

        #self.conv2 = torch.nn.Conv1d(out_ch, 
        #                             out_ch, 
        #                             kernel_size=1, 
        #                             stride=1)
        #self.act2 = get_activation(act_type, out_ch)

        self.res_conv = torch.nn.Conv1d(in_ch, 
                                        out_ch, 
                                        kernel_size=1, 
                                        stride=stride)

    def forward(self, x):
        x_res = x # store input for later
        
        # -- first section --
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.act1(x)

        # -- second section --
        #x = self.conv2(x)
        #if self.norm_type is not None:
        #    x = self.norm2(x)
        #x = self.act2(x)

        # -- residual connection --
        x_res = self.res_conv(x_res)
        if self.res_norm is not None:
            x_res = self.res_norm(x_res)

        return x + x_res

class dsTCNModel(Base):
    """ Downsampling Temporal convolutional network.

        Args:
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 15
            stride (int): Stide size when applying convolutional filter. Default: 2 
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 8
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 1
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 32
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
            causal (bool): Causal TCN configuration does not consider future input values. Default: False
            skip_connections (bool): Skip connections from each block to the output. Default: False
            norm_type (str): Type of normalization layer to use 'BatchNorm', 'LayerNorm', 'InstanceNorm'. Default: None
    """
    def __init__(self, 
                 ninputs=1,
                 noutputs=2,
                 nblocks=10, 
                 kernel_size=3, 
                 stride=2,
                 dilation_growth=8, 
                 channel_growth=1, 
                 channel_width=32, 
                 stack_size=4,
                 grouped=False,
                 causal=False,
                 skip_connections=False,
                 norm_type='BatchNorm',
                 act_type='PReLU',
                 **kwargs):
        super(dsTCNModel, self).__init__()
        self.save_hyperparameters()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = ninputs if n == 0 else out_ch 
            out_ch = channel_width if n == 0 else in_ch + channel_growth
            dilation = dilation_growth ** (n % stack_size)

            self.blocks.append(dsTCNBlock(
                in_ch, 
                out_ch,
                kernel_size,
                stride,
                dilation,
                norm_type,
                act_type
            ))

        self.output = torch.nn.Conv1d(out_ch, 
                                      noutputs, 
                                      kernel_size=1)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        
        x = self.output(x)
        #x = torch.sigmoid(x)

        return x

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = 0
        for n in range(self.hparams.nblocks):
            rf += (self.hparams.kernel_size - 1) * \
                  (self.hparams.nblocks * self.hparams.stride)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--patience', type=int, default=40)
        # --- model related ---
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=2)
        parser.add_argument('--nblocks', type=int, default=8)
        parser.add_argument('--kernel_size', type=int, default=15)
        parser.add_argument('--stride', type=int, default=2)
        parser.add_argument('--dilation_growth', type=int, default=8)
        parser.add_argument('--channel_growth', type=int, default=1)
        parser.add_argument('--channel_width', type=int, default=32)
        parser.add_argument('--stack_size', type=int, default=4)
        parser.add_argument('--grouped', default=False, action='store_true')
        parser.add_argument('--causal', default=False, action="store_true")
        parser.add_argument('--skip_connections', default=False, action="store_true")
        parser.add_argument('--norm_type', type=str, default='BatchNorm')
        parser.add_argument('--act_type', type=str, default='PReLU')

        return parser