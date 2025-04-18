import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

# from .resnet import resnet34

"""
provide three models:
    I2U_Net_L
    I2U_Net_M
    I2U_Net_S
"""

nonlinearity = partial(F.relu, inplace=True)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def BNReLU(num_features):
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU()
    )


# ############################################## drop block ###########################################

class Drop(nn.Module):
    # drop_rate : 1-keep_prob  (all droped feature points)
    # block_size :
    def __init__(self, drop_rate=0.1, block_size=2):
        super(Drop, self).__init__()

        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):

        if not self.training:
            return x

        if self.drop_rate == 0:
            return x

        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)
        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


# ############################################## HIFA_module_v1 ###########################################

class SPP_inception_block(nn.Module):
    def __init__(self, in_channels):
        super(SPP_inception_block, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # [3, 3]
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # [2, 2]
        # self.pool = nn.MaxPool2d(kernel_size=[4, 4], stride=4) # [1, 1]
        # self.pool = nn.MaxPool2d(kernel_size=[1, 1], stride=2) # [4, 4]
        # self.pool = nn.MaxPool2d(kernel_size=[1, 1], stride=1)   # [7, 7]
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()  # [4, 256, 7, 7]
        pool_1 = self.pool1(x).view(b, c, -1)  # [2, 256, 3, 3], [2, 256, 9]
        # pool_1 = self.pool(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)  # [2, 256, 2, 2], [2, 256, 4]
        pool_3 = self.pool3(x).view(b, c, -1)  # [2, 256, 1, 1], [2, 256, 1]
        pool_4 = self.pool4(x).view(b, c, -1)  # [2, 256, 1, 1], [2, 256, 1]

        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # [2, 256, 15]

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(
            self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))  # self.conv1x1 is not necessary

        cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out  # [2, 256, 7, 7]
        cnn_out = cnn_out.view(b, c, -1)  # [2, 256, 49]

        out = torch.cat([pool_cat, cnn_out], -1)  # [2, 256, 64]
        out = out.permute(0, 2, 1)  # [2, 64, 256]

        return out


class NonLocal_spp_inception_block(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels=512, ratio=2):
        super(NonLocal_spp_inception_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = in_channels // ratio
        self.value_channels = in_channels // ratio

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            BNReLU(self.key_channels),
        )

        self.f_query = self.f_key

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.spp_inception_v = SPP_inception_block(self.key_channels)
        self.spp_inception_k = SPP_inception_block(self.key_channels)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)  # [2, 512, 7, 7]

        x_v = self.f_value(x)  # [2, 256, 7, 7]
        value = self.spp_inception_v(x_v)  # [2, 64, 256]  15+49

        query = self.f_query(x).view(batch_size, self.key_channels, -1)  # [2, 256, 7, 7], [2, 256, 49]
        query = query.permute(0, 2, 1)  # [2, 49, 256]

        x_k = self.f_key(x)  # [2, 256, 7, 7]
        key = self.spp_inception_k(x_k)  # [2, 64, 256]  15+49
        key = key.permute(0, 2, 1)  # # [2, 256, 64]

        sim_map = torch.matmul(query, key)  # [2, 49, 64]
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)  # [2, 49, 256]
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])  # [4, 256, 7, 7]
        context = self.W(context)  # [4, 512, 7, 7]

        return context


class HIFA_V1(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels=512, ratio=2, dropout=0.0):
        super(HIFA_V1, self).__init__()

        self.NSIB = NonLocal_spp_inception_block(in_channels=in_channels, ratio=ratio)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            BNReLU(in_channels)
            # nn.Dropout2d(dropout)
        )

    def forward(self, feats):
        att = self.NSIB(feats)
        output = self.conv_bn_dropout(torch.cat([att, feats], 1))

        return output


# ############################################## HIFA_module_v2 ############################################################

class SPP_inception_block_v2(nn.Module):
    def __init__(self, in_channels):
        super(SPP_inception_block_v2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[1, 1], stride=2)  # [4, 4]
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # [3, 3]
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # [2, 2]
        self.pool4 = nn.MaxPool2d(kernel_size=[4, 4], stride=4)  # [1, 1]

        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()  # [4, 272, 7, 7]
        pool_1 = self.pool1(x).view(b, c, -1)  # [2, 272, 4, 4], [2, 272, 16]
        # pool_1 = self.pool(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)  # [2, 272, 3, 3], [2, 272, 9]
        pool_3 = self.pool3(x).view(b, c, -1)  # [2, 272, 2, 2], [2, 272, 4]
        pool_4 = self.pool4(x).view(b, c, -1)  # [2, 272, 1, 1], [2, 272, 1]

        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # [2, 272, 30]

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate4_out = nonlinearity(self.dilate4(x))  # self.conv1x1 is not necessary

        cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out  # [2, 272, 7, 7]
        cnn_out = cnn_out.view(b, c, -1)  # [2, 272, 49]

        out = torch.cat([pool_cat, cnn_out], -1)  # [2, 272, 79]
        out = out.permute(0, 2, 1)  # [2, 79, 256]

        return out


class NonLocal_spp_inception_block_v2(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels=512, ratio=2):
        super(NonLocal_spp_inception_block_v2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.value_channels = in_channels // ratio  # key == value
        self.query_channels = in_channels // ratio

        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1,
                      padding=0),
            BNReLU(self.value_channels),
        )

        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.query_channels, kernel_size=1, stride=1,
                      padding=0),
            BNReLU(self.query_channels),
        )

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.spp_inception_v = SPP_inception_block_v2(self.value_channels)  # key == value
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)  # [4, 544, 7, 7]

        x_v = self.f_value(x)  # [4, 272, 7, 7]
        value = self.spp_inception_v(x_v)  # [4, 79, 272]  30+49

        query = self.f_query(x).view(batch_size, self.value_channels, -1)  # [4, 272, 7, 7], [4, 272, 49]
        query = query.permute(0, 2, 1)  # [4, 49, 272]

        key_0 = value
        key = key_0.permute(0, 2, 1)  # [4, 272, 79]

        sim_map = torch.matmul(query, key)  # [4, 49, 79]
        sim_map = (self.value_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)  # [4, 49, 272]
        context = context.permute(0, 2, 1).contiguous()  # [4, 272, 49]
        context = context.view(batch_size, self.value_channels, *x.size()[2:])  # [4, 272, 7, 7]
        context = self.W(context)  # [4, 544, 7, 7]

        return context


class HIFA_V2(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels=512, ratio=2, dropout=0.0):
        super(HIFA_V2, self).__init__()

        self.NSIB = NonLocal_spp_inception_block_v2(in_channels=in_channels, ratio=ratio)
        # def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, psp_size=(1,3,6,8)):

    def forward(self, feats):
        att = self.NSIB(feats)
        output = att + feats

        return output


if __name__ == '__main__':
    input = torch.rand(2, 3, 100, 100)
    model = HIFA_V2(in_channels=3, ratio=2)
    out12 = model(input)
    print(out12.shape)