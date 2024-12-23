import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResnetBlock3d(nn.Module):
    """Define a 3dResnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the 3dResnet block
        """
        super(ResnetBlock3d, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        #print("##########print x and conv_block x size")
        #print(x.size())
        #print(self.conv_block(x).size())

        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Mila(nn.Module):
    def forward(self,x):
        return x * tf.math.tanh(tf.math.softplus(-1.0 + x))


class IonResnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer = nn.BatchNorm3d, use_dropout = False, n_blocks = 6):
        """Construct a Resnet-based network

        Parameters:
            input_nc (int)      -- the number of channels of input, defualt is two, including charge and radius of atoms
            output_nc (int)     -- the number of channels of output
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(IonResnet, self).__init__()


        model = [nn.ReflectionPad3d(1),
                 nn.Conv3d(input_nc, ngf, kernel_size = 3, stride = 1, padding = 0, bias = True),
                 norm_layer(ngf),
                 nn.ReLU(True)]
 
        mult = 1
        model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size = 3, stride = 2, padding = 1, bias=True),
                  norm_layer(ngf * mult * 2),
                  nn.ReLU(True)]


        for i in range(n_blocks):
            model += [ResnetBlock3d(ngf * mult * 2, padding_type = 'reflect', norm_layer = norm_layer, use_dropout = use_dropout, use_bias= True)]

        mult = 1
        model += [nn.ConvTranspose3d(ngf*mult*2, ngf*mult, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = True),
        norm_layer(ngf*mult),
        nn.ReLU(True)]
      

        
        model += [nn.ReflectionPad3d(1)]
        model += [nn.Conv3d(ngf*mult, output_nc, kernel_size = 3, padding = 0)]
        model += [nn.ReLU(True)]



        self.model = nn.Sequential(*model)

    def forward(self,input):
        return self.model(input)



