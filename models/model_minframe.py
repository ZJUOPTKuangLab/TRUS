import torch.nn as nn
import torch.nn.functional as F
import torch
from inspect import isfunction
from models.SpatialTrans import SpatioTransLayer
import torch
from torch import nn
from torch.nn import functional as F
from inspect import isfunction
from einops import rearrange

# 上采样+拼接
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        '''
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        '''
        super(Up, self).__init__()
        if bilinear:
            # 双线性差值
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) # 拼接后为1024，经历第一个卷积后512
        else:
            # 转置卷积实现上采样
            # 输出通道数减半，宽高增加一倍
            self.up = nn.ConvTranspose2d(in_channels,out_channels//2,kernel_size=2,stride=2)
            self.conv = doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        # 上采样
        x1 = self.up(x1)
        # 拼接
        x = torch.cat([x1,x2],dim=1)
        # 经历双卷积
        x = self.conv(x)
        return x

# 双卷积层
def doubleConv(in_channels,out_channels,mid_channels=None):
    '''
    :param in_channels: 输入通道数
    :param out_channels: 双卷积后输出的通道数
    :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
    :return:
    '''
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)

# 下采样
def down(in_channels,out_channels):
    # 池化 + 双卷积
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels,out_channels))
    return nn.Sequential(*layer)

# 整个网络架构
class Unet(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,base_channel=64):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.maxpool=nn.MaxPool2d(2,stride=2)
        # 输入
        self.in_conv = doubleConv(self.in_channels,base_channel)
        self.in_conv1 = doubleConv(base_channel*2, base_channel)
        # self.cat=torch.cat([x1,x2],dim=1)
        # 下采样
        self.down1 = down(base_channel,base_channel*2) # 64,128
        self.down2 = down(base_channel*2,base_channel*4) # 128,256
        self.down3 = down(base_channel*4,base_channel*8) # 256,512
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
        factor = 2  if self.bilinear else 1
        self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512

        # 上采样 + 拼接
        self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) # 1024(双卷积的输入),256（双卷积的输出）
        self.up2 = Up(base_channel*8 ,base_channel*4 // factor,self.bilinear)
        self.up3 = Up(base_channel*4 ,base_channel*2 // factor,self.bilinear)
        self.up4 = Up(base_channel*2 ,base_channel*4,self.bilinear)
        # 输出
        self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

        self.output=nn.Sigmoid()

    def forward(self,x,x_w):
        x0 = self.in_conv(x)
        # y0 = self.in_conv(y)
        x_w0=self.in_conv(x_w)
        x1_1=torch.cat([x0,x_w0],dim=1)
        x1=self.in_conv1(x1_1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 不要忘记拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x=nn.functional.pixel_shuffle(x,2)
        out = self.out(x)
        out=self.output(out)

        return out

class ResUnet(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,base_channel=64):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(ResUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.maxpool=nn.MaxPool2d(2,stride=2)
        # 输入
        self.in_conv = doubleConv(self.in_channels,base_channel)
        self.in_conv1 = doubleConv(base_channel*2, base_channel)
        # self.cat=torch.cat([x1,x2],dim=1)
        # 下采样
        self.down1 = down(base_channel,base_channel*2) # 64,128
        self.down2 = down(base_channel*2,base_channel*4) # 128,256
        self.down3 = down(base_channel*4,base_channel*8) # 256,512
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
        factor = 2  if self.bilinear else 1
        self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512

        # 上采样 + 拼接
        self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) # 1024(双卷积的输入),256（双卷积的输出）
        self.up2 = Up(base_channel*8 ,base_channel*4 // factor,self.bilinear)
        self.up3 = Up(base_channel*4 ,base_channel*2 // factor,self.bilinear)
        self.up4 = Up(base_channel*2 ,base_channel*4,self.bilinear)
        # 输出
        self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

        self.output=nn.Sigmoid()

    def forward(self,x,x_w):
        x0 = self.in_conv(x)
        # y0 = self.in_conv(y)
        x_w0=self.in_conv(x_w)
        x1_1=torch.cat([x0,x_w0],dim=1)
        x1=self.in_conv1(x1_1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 不要忘记拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x=nn.functional.pixel_shuffle(x,2)
        out = self.out(x)
        out=self.output(out)

        return {'out': out}

class MainFrame(nn.Module):
    def __init__(
            self,
            img_dim,
            in_channel,
            f_maps=[64, 128, 256, 512],
            input_dropout_rate=0.1,
            num_layers=0
    ):
        super(MainFrame, self).__init__()
        self.img_dim = img_dim
        self.f_maps = f_maps
        # 2Conv + Down
        self.encoders = self.temporalSqueeze(
            f_maps=[in_channel] + f_maps
        )

        # up + 2Conv
        self.decoders = self.temporalExcitation(
            f_maps=f_maps[::-1] + [in_channel]
        )
        # self.final_conv = nn.Conv3d(1, 1, 1)
        self.pixel_conv=nn.Conv2d(in_channel,in_channel*64,kernel_size=3,padding=1,bias=False)
        self.double_conv = doubleConv(in_channel, out_channels=in_channel)
        self.out_conv2 = nn.Conv2d(in_channel, out_channels=1, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channel, out_channels=1, kernel_size=1)
        self.output = nn.Sigmoid()

        self.up_sample = nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(4, 4),
                                        stride=(2, 2), padding=(1, 1))
    def temporalSqueeze(self, f_maps, num_layers=0):
        model_list = nn.ModuleList([])

        for idx in range(1, len(f_maps)):
            encoder_layer = SqueezeLayer(
                in_channels=f_maps[idx - 1],
                out_channels=f_maps[idx],
            )
            model_list.append(encoder_layer)
        return model_list

    def temporalExcitation(self, f_maps):
        model_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            decoder_layer = ExcitationLayer(
                in_channels=f_maps[idx - 1],
                out_channels=f_maps[idx],
                if_up_sample=True
            )
            model_list.append(decoder_layer)
        return model_list

    def process_by_trans(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            before_down, x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, before_down)

        x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)
        ##the upsample version(latest)
        # x=  self.pixel_conv(x)
        # x = nn.functional.pixel_shuffle(x, 2)
        x = self.up_sample(x)
        x = self.double_conv(x)
        x = self.out_conv2(x)
        ##no upsample version(03.10)
        # x = self.out_conv(x)
        ##
        x = self.output(x)


        return x


class SqueezeLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3
    ):
        super(SqueezeLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=True
        )
        self.down_sample = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))

    def forward(self, x):
        before_down = self.conv_net(x)
        x = self.down_sample(before_down)
        return before_down, x


class ExcitationLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_up_sample=True,
            kernel_size=3,
    ):
        super(ExcitationLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=False
        )
        self.if_up_sample = if_up_sample
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(4, 4),
                                            stride=(2, 2), padding=(1, 1))

    def forward(self, x, encoder_features):
        if self.if_up_sample:
            x = self.up_sample(x)
        x += encoder_features
        # x = torch.cat((encoder_features, x), dim=2)
        x = self.conv_net(x)
        return x


class SingleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1
    ):
        super(SingleConv, self).__init__()
        self.add_module('Conv2d',
                        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        self.add_module('batch_norm',nn.BatchNorm2d(out_channels))
        self.add_module('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_encoder,
            kernel_size=3
    ):
        super(DoubleConv, self).__init__()
        if if_encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, padding=1))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, padding=1))


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_layers,
            num_heads,
            window_size,
            hidden_dim,
            attn_drop=0.,
            input_drop=0.,
    ):
        super(SpatioTransformer, self).__init__()
        self.transformer = SpatioTransLayer(
            dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=hidden_dim/embedding_dim,
            qkv_bias=True,
            qk_scale=None,
            drop=input_drop,
            attn_drop=attn_drop,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer(x, H, W)
        x = rearrange(x, 'b (h p2) c -> b c h p2', p2=W)
        return x

class Transformer(MainFrame):
    def __init__(
            self,
            img_dim,
            in_channel,
            embedding_dim,
            window_size,
            num_heads,
            hidden_dim,
            num_transBlock,
            attn_dropout_rate,
            f_maps=[128, 256, 512],
            input_dropout_rate=0.1
    ):
        super(Transformer, self).__init__(
            img_dim,
            in_channel,
            f_maps=f_maps,
            input_dropout_rate=input_dropout_rate
        )
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim

        self.conv_before_trans = SingleConv(
            f_maps[-1],
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_after_trans = SingleConv(
            embedding_dim,
            f_maps[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.layers = nn.ModuleList([])
        for idx in range(num_transBlock):
            self.layers.append(
                SpatioTransformer(
                    embedding_dim=embedding_dim,
                    num_layers=2,
                    num_heads=num_heads,
                    window_size=window_size,
                    hidden_dim=hidden_dim,
                    attn_drop=attn_dropout_rate,
                    input_drop=input_dropout_rate
                )
            )


    def process_by_trans(self, x):
        x = self.conv_before_trans(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_after_trans(x)

        return x