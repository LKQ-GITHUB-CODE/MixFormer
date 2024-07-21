import torch
'''device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(1))
print(torch.cuda.device_count())'''
from einops import rearrange
import torch
from torch import nn
from utils import *

from einops.layers.torch import Rearrange
#spatial_attention
class CA_Block(nn.Module):
    def __init__(self, channel, h, w,reduction=16):
        super(CA_Block, self).__init__()

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

#channel_spatial
class Channel_Attention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(channel // ratio, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class MMDA(nn.Module):
    def __init__(self, channel, ratio=8,h=56, w=56,emdim=96):
        super(MMDA, self).__init__()
        self.h=h
        self.Spatial_Attention=CA_Block(channel,h, w)
        self.Channel_Attention=Channel_Attention(channel,ratio)
        self.Global_Attention= BasicLayer(dim=channel, input_resolution=(h,w), depth=2, num_heads=4,
                                          window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                          drop=0, attn_drop=0, norm_layer=nn.LayerNorm,downsample=None)
    def forward(self, x):

        Spatial_att=self.Spatial_Attention(x)
        Channel_Att=self.Channel_Attention(x)

        y = Rearrange('b c h w -> b (h w) c')(x)
        Global_Att=self.Global_Attention(y)
        Global_Att=rearrange(Global_Att, ' b (h w) c -> b c h w', h=self.h)
        out = Global_Att + Spatial_att * Channel_Att
        return out



'''if __name__ == '__main__':
    x = torch.randn(1, 256, 56, 56)  # b, c, h, w
    ca_model = MMDA(channel=256,ratio=8)
    y = ca_model(x)
    print(y.shape)'''