import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
class UNet_MGS_base(nn.Module):
    def __init__(self, n_channels, n_classes,chs=(16, 32, 64, 128, 256, 128, 64, 32, 16), project_dim=64):
        super(UNet_MGS_base, self).__init__()
        self.n_channels = n_channels
        self.inc = DoubleConv2d(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])
        self.up1 = Up(chs[4] + chs[3], chs[5])
        self.up2 = Up(chs[5] + chs[2], chs[6])
        self.up3 = Up(chs[6] + chs[1], chs[7])
        self.up4 = Up(chs[7] + chs[0], chs[8])


        self.convds3 = nn.Conv2d(chs[3], n_classes, kernel_size=1)
        self.convds2 = nn.Conv2d(chs[2], n_classes, kernel_size=1)
        self.convds1 = nn.Conv2d(chs[1], n_classes, kernel_size=1)
        self.convds0 = nn.Conv2d(chs[0], n_classes, kernel_size=1)

        self.projector = nn.Sequential(
            nn.Conv2d(chs[2], chs[2], kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(chs[2], project_dim, kernel_size=1)
        )

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        # print(x.size())
        H, W = x.size()[2], x.size()[3]
        new_H = ((H + 15) // 16) * 16
        new_W = ((W + 15) // 16) * 16
        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        # forward
        x0 = self.inc(x)  
        x1 = self.down1(x0) 
        x2 = self.down2(x1) 
        x3 = self.down3(x2)  
        x4 = self.down4(x3)
        x3_1 = self.up1(x4, x3) 
        x2_2 = self.up2(x3_1, x2)
        x1_3 = self.up3(x2_2, x1) 
        x0_4 = self.up4(x1_3, x0) 

        out = dict()
        out['project'] = self.projector(x2_2) 
        out['project_map'] = F.interpolate(self.convds0(x0_4),size=x2_2.shape[2:], mode='bilinear', align_corners=False) 
        out['level3'] = F.interpolate(self.convds3(x3_1), size=(H, W), mode='bilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=(H, W), mode='bilinear', align_corners=False) 
        out['level1'] = F.interpolate(self.convds1(x1_3), size=(H, W), mode='bilinear', align_corners=False) 
        out['out'] = F.interpolate(self.convds0(x0_4), size=(H, W), mode='bilinear', align_corners=False) 
        out['decode_1']=F.interpolate(x0, size=(H, W), mode='bilinear', align_corners=False) 
        return out
    
class UNet_MGS(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16), project_dim=64):
        super(UNet_MGS, self).__init__()
        self.unet = UNet_MGS_base(n_channels=n_channels,n_classes=n_classes, chs=depth, project_dim=project_dim)

    def forward(self, x):
        x = self.unet(x)
        return x
