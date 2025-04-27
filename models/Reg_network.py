import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.STN import SpatialTransformer_2d, Re_SpatialTransformer_2d

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
# SE
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #
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
    

class PAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(PAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // ratio
        

        self.gamma = nn.Parameter(torch.zeros(1))
        

        self.conv_b = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_c = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size, height, width = x.size()[0], x.size()[2], x.size()[3]

        b = self.conv_b(x)
        c = self.conv_c(x)
        d = self.conv_d(x)

        # Reshape and transpose
        vec_b = b.view(batch_size, self.inter_channels, -1) # b*(c/r)*(h*w)
        vec_c = c.view(batch_size, self.inter_channels, -1).permute(0, 2, 1) # b*(h*w)*(c/r)
        vec_d = d.view(batch_size, self.in_channels, -1).permute(0, 2, 1) #b*(h*w)*(c)

 
        bcT = torch.bmm(vec_c, vec_b) #b*(h*w)*(h*w)
        attention = F.softmax(bcT, dim=-1) #b*(h*w)*(h*w) 
        

        bcTd = torch.bmm(attention, vec_d).permute(0, 2, 1)
        bcTd = bcTd.view(batch_size, self.in_channels, height, width)

        out = self.gamma * bcTd + x
        return out
    
# unet + se + pam 
class Up_se_pam(nn.Module):
    def __init__(self, in_channels, out_channels,se_reduction=16,pam_ratio=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv2d(in_channels, out_channels)
        self.se = SELayer(in_channels,se_reduction)
        self.pam = PAM(in_channels,pam_ratio)

    def forward(self, x1, x2):
        x1 = self.up(x1) #
        x = torch.cat([x2, x1], dim=1)
        x = self.se(x) #
        x = self.pam(x) # 
        return self.conv(x)

class Up_se(nn.Module):
    def __init__(self, in_channels, out_channels,se_reduction=16):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv2d(in_channels, out_channels)
        self.se = SELayer(in_channels,se_reduction)

    def forward(self, x1, x2):
        x1 = self.up(x1) #
        x = torch.cat([x2, x1], dim=1)
        x = self.se(x) # 
        return self.conv(x)   
     
class UNet_se_2pam(nn.Module):
    def __init__(self, n_channels, chs=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_se_2pam, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv2d(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])
        self.up1 = Up_se_pam(chs[4] + chs[3], chs[5],16,8) 
        self.up2 = Up_se_pam(chs[5] + chs[2], chs[6],16,8)
        self.up3 = Up_se(chs[6] + chs[1], chs[7],16)
        self.up4 = Up(chs[7] + chs[0], chs[8])
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        # print(x.size())
        H = x.size()[2]
        W = x.size()[3]

        diffH = (16 - H % 16) % 16
        diffW = (16 - W % 16) % 16

        x = F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x1 = self.inc(x) 
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x[:, :, :H,:W]
class UNet_reg_with_se_2pam(nn.Module):
    def __init__(self, n_channels=1, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_reg_with_se_2pam, self).__init__()
        self.unet = UNet_se_2pam(n_channels=n_channels*2, chs=depth)
        self.out_conv = nn.Conv2d(depth[-1], 2, 1)

        self.stn = SpatialTransformer_2d(islabel=False)
        self.lstn = SpatialTransformer_2d(islabel=True)
        self.rstn = Re_SpatialTransformer_2d(islabel=False)
        self.lrstn = SpatialTransformer_2d(islabel=True)

    def forward(self, moving, fixed, mov_label=None, fix_label=None):
        x = torch.cat([fixed, moving], dim=1)
        x = self.unet(x)
        flow = self.out_conv(x)
        w_m_to_f = self.stn(moving, flow,mode = "bilinear")

        w_f_to_m = self.rstn(fixed, flow)

        if mov_label is not None:
            w_label_m_to_f = self.lstn(mov_label, flow, mode = "nearest")
        else:
            w_label_m_to_f = None

        if fix_label is not None:
            w_label_f_to_m = self.lrstn(fix_label, flow,mode = "nearest")
        else:
            w_label_f_to_m = None

        return w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow

