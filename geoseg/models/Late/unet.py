import torch
import torch.nn as nn
import copy

import numpy as np
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) #反卷积，转置卷积

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1) #上采样
        #input is CHW
        x = torch.cat([x2, x1], dim=1) #特征整合
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels*13, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.reshape(B, C*T, H, W)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        x = x.to(device)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
class LSTMUNet(nn.Module):
    def __init__(self, sequence_length, in_channels, img_height, img_width, n_label, lstm_hidden_size=64):
        super(LSTMUNet, self).__init__()
        input_size = 3 * 256 * 256
        lstm_hidden_size = 512
        # LSTM层
        #config = ModelConfig()
        self.lstm1 = UNet()
        self.lstm2 = UNet()
        # U-Net架构
      
    def forward(self, x):
       
        
        
        BS =  x.shape[0]  
      #  print(x.shape)
        stacked_tensor1 = torch.empty(13,BS,3, 256, 256)
        stacked_tensor2 = torch.empty(13,BS,3, 256, 256)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
        stacked_tensor1.to(device)
        stacked_tensor2.to(device)
        for i in range(10):
             n1 = x[:, 2*i:2*i+1, :, :, :]
             n2 = x[:, 2*i+1:2*i+2, :, :, :]
           #  print(n1.shape)
           #  n1 = n2/2+n1/2
             n1 = n1.reshape(BS,3,256,256)
             n2 = n2.reshape(BS,3,256,256)
             
            
             #unet_out2 = self.unet2(n2)
            # unet_out = unet_out1/2+unet_out2/2
             #unet_out = n2
             stacked_tensor1[i] = n1
             stacked_tensor2[i] = n2
        stacked_tensor1 = stacked_tensor1.permute(1,0,2,3,4)
        stacked_tensor2 = stacked_tensor2.permute(1,0,2,3,4)
        lstm_out1 = self.lstm1(stacked_tensor1)
        lstm_out2 = self.lstm2(stacked_tensor2)
  
        return     (lstm_out1+lstm_out2)/2      

 
  
