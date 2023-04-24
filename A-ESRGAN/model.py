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
import torchvision.models as models
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from math import log2, ceil
from torch.nn import functional as F
from collections import OrderedDict
from torch.nn import init as init
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = [
    "ContentLoss","uNetDiscriminatorAesrgan", "gen_rpa2x", "gen_rrdb2x", "bsrgan_x2", "content_loss",

]


def unshuffle(x, scale=1):
    if scale == 1:
        return x
    b, c, h, w = x.size()
    h //= scale
    w //= scale
    num_ch = c * (scale**2)
    return x.view(b, c, h, scale, w, scale).permute(0, 1, 3, 5, 2, 4).reshape(b, num_ch, h, w)


class rrdb_block(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(rrdb_block, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialize layer weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            init.kaiming_normal_(layer.weight)
            # scale factor emperically set to 0.1
            layer.weight.data *= 0.1

    def forward(self, x):
        z1 = self.lrelu(self.conv1(x))
        z2 = self.lrelu(self.conv2(torch.cat((x, z1), 1)))
        z3 = self.lrelu(self.conv3(torch.cat((x, z1, z2), 1)))
        z4 = self.lrelu(self.conv4(torch.cat((x, z1, z2, z3), 1)))
        z5 = self.conv5(torch.cat((x, z1, z2, z3, z4), 1))
        # scale factor set to 0.2, according to ESRGAN spec
        return z5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdblk1 = rrdb_block(num_feat, num_grow_ch)
        self.rdblk2 = rrdb_block(num_feat, num_grow_ch)
        self.rdblk3 = rrdb_block(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdblk1(x)
        out = self.rdblk2(out)
        out = self.rdblk3(out)
        return out * 0.2 + x


class US(nn.Module):
    """Up-sampling block
    """

    def __init__(self, num_feat, scale):
        super(US, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 1)
        # plugin pixel attention
        self.pa_conv = nn.Conv2d(num_feat, num_feat, 1)
        self.pa_sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_ = self.conv1(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
        x_ = self.lrelu(x_)
        z = self.pa_conv(x_)
        z = self.pa_sigmoid(z)
        z = torch.mul(x_, z) + x_
        z = self.conv2(z)
        out = self.lrelu(z)
        return out


class RPA(nn.Module):
    """Residual pixel-attention block
    """

    def __init__(self, num_feat):
        super(RPA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 1)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialize layer weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv3]:
            init.kaiming_normal_(layer.weight)
            # scale factor emperically set to 0.1
            layer.weight.data *= 0.1

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        z = self.conv3(z)
        z = self.sigmoid(z)
        z = x * z + x
        z = self.conv4(z)
        out = self.lrelu(z)
        return out


@ARCH_REGISTRY.register()
class Generator_RPA(nn.Module):
    """The generator of A-ESRGAN is comprised of residual pixel-attention(PA) blocks
     and consequent up-sampling blocks.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=20):
        super(Generator_RPA, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # residual pixel-attention blocks
        self.rpa = nn.Sequential(
            OrderedDict(
                [("rpa{}".format(i), RPA(num_feat=num_feat)) for i in range(num_block)]))
        # up-sampling blocks with pixel-attention
        num_usblock = ceil(log2(scale))
        self.us = nn.Sequential(
            OrderedDict(
                [("us{}".format(i), US(num_feat=num_feat, scale=2)) for i in range(num_usblock)]))
        self.conv2 = nn.Conv2d(num_feat, num_feat // 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat // 2, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z_ = self.rpa(z)
        z = z + z_
        z = self.us(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        out = self.conv3(z)
        return out


class Generator_RRDB(nn.Module):
    """The generator of A-ESRGAN is comprised of Residual in Residual Dense Blocks(RRDBs) as
    ESRGAN. And we employ pixel unshuffle to input feature before the network.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(Generator_RRDB, self).__init__()
        self.scale = scale
        num_in_ch *= 16 // (scale)**2
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # embed rrdb network here
        self.rrdb = nn.Sequential(
            OrderedDict(
                [("rrdb{}".format(i), RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch)) for i in range(num_block)]))
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # conv3 & conv4 are for up-sampling
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv6 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        z = unshuffle(x, scale=4 // self.scale)
        z = self.conv1(z)
        z_ = self.conv2(self.rrdb(z))
        z = z + z_
        z = self.lrelu(self.conv3(F.interpolate(z, scale_factor=2, mode='nearest')))
        z = self.lrelu(self.conv4(F.interpolate(z, scale_factor=2, mode='nearest')))
        z = self.conv6(self.lrelu(self.conv5(z)))
        return z


def gen_rrdb2x() -> Generator_RRDB:
    print("* Generator_RRDB 2x")
    model = Generator_RRDB(scale=2)

    return model

def gen_rpa2x() -> Generator_RPA:
    print("* Generator_RPA 2x")
    model = Generator_RPA(scale=2)

    return model

##################################   



# @ARCH_REGISTRY.register()
class add_attn(nn.Module):

    def __init__(self, x_channels, g_channels=256):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels, x_channels, kernel_size=2, stride=2, padding=0, bias=False)

        self.phi = nn.Conv2d(g_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(x_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, sigm_psi_f


# @ARCH_REGISTRY.register()
class unetCat(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        norm = spectral_norm
        self.convU = norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False))

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(input_2, scale_factor=2, mode='bilinear', align_corners=False)

        output_2 = F.leaky_relu(self.convU(input_2), negative_slope=0.2, inplace=True)

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y


# @ARCH_REGISTRY.register()
class UNetDiscriminatorAesrgan(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorAesrgan, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=False))

        # Center
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, 2, 1, bias=False))

        self.gating = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False))

        # attention Blocks
        self.attn_1 = add_attn(x_channels=num_feat * 4, g_channels=num_feat * 4)
        self.attn_2 = add_attn(x_channels=num_feat * 2, g_channels=num_feat * 4)
        self.attn_3 = add_attn(x_channels=num_feat, g_channels=num_feat * 4)

        # Cat
        self.cat_1 = unetCat(dim_in=num_feat * 8, dim_out=num_feat * 4)
        self.cat_2 = unetCat(dim_in=num_feat * 4, dim_out=num_feat * 2)
        self.cat_3 = unetCat(dim_in=num_feat * 2, dim_out=num_feat)

        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        gated = F.leaky_relu(self.gating(x3), negative_slope=0.2, inplace=True)

        # Attention
        attn1, self.ly1 = self.attn_1(x2, gated)
        attn2, self.ly2 = self.attn_2(x1, gated)
        attn3, self.ly3 = self.attn_3(x0, gated)

        # upsample
        x3 = self.cat_1(attn1, x3)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        x4 = self.cat_2(attn2, x4)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        x5 = self.cat_3(attn3, x5)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
    
    def getAttentionLayers(self):
        return self.ly1, self.ly2, self.ly3


def uNetDiscriminatorAesrgan() -> UNetDiscriminatorAesrgan:
    print("* UNetDiscriminatorAesrgan")
    model = UNetDiscriminatorAesrgan(3)

    return model



##################################   

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_nodes: list,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_nodes = feature_model_extractor_nodes
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, feature_model_extractor_nodes)
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> list[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device
        losses = []

        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        for i in range(len(self.feature_model_extractor_nodes)):
            losses.append(F.l1_loss(sr_feature[self.feature_model_extractor_nodes[i]],
                                          gt_feature[self.feature_model_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device=device)

        return losses

def content_loss(**kwargs) -> ContentLoss:
    content_loss_model = ContentLoss(**kwargs)

    return content_loss_model


##################################  

class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out

class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out
    
class BSRGAN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23,
            upscale_factor: int = 4,
    ) -> None:
        super(BSRGAN, self).__init__()
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling1 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        if upscale_factor == 4:
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all model layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        if self.upscale_factor == 4:
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))

        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def bsrgan_x2(**kwargs: Any) -> BSRGAN:
    print("* BSRGAN 2x")
    model = BSRGAN(upscale_factor=2, **kwargs)

    return model