import torch
from torch import nn

        
from openstl.modules import (ConvSC,ConvMYP,TAUSubBlock, MogaSubBlock,SimTASubBlock)

import globalss

from PIL import Image 
import matplotlib.pyplot as plt
import os  
import numpy as np  
import matplotlib.pyplot as plt  
from PIL import Image  

def show(feature, save_path=None,num =None):  
    # 将每个特征图从 GPU 转换为 CPU 并转为 numpy 数组  
    feature_map_data_list = [feature[0, 2].detach().cpu().numpy() ]  
    
    print(feature.shape)  
    nAddss = globalss.print_global_variable()  
    addss = str(nAddss)  

    # 生成要保存的文件夹路径  
    save_dir = '/data/pth/liangjiaxuan-pth/testCam/My/'  
    os.makedirs(save_dir, exist_ok=True)  
    imhs = os.path.join(save_dir, f"{addss}.png")  
    OImg = '/data/pth/liangjiaxuan-pth/20231231Datas/ThLabel/00' + addss + "_M0.png"  
    Imgs = Image.open(OImg)  
    print(imhs)
    # 创建文件夹（如果它不存在）  

    print(feature.shape)
    # 获取原始图像的尺寸  
    original_width, original_height = Imgs.size  

    # 可视化每个特征图的热力图  
    #plt.figure(figsize=(original_width / 100, original_height / 100))  # 根据原始图像尺寸设置绘图大小  
    
    for i, feature_map_data in enumerate(feature_map_data_list):  
      
     
        plt.subplot(1, len(feature_map_data_list), i + 1)  
        plt.imshow(feature_map_data, cmap="jet", alpha=0.9)  # 使用 jet 颜色映射  
        plt.imshow(Imgs, alpha=0.1)  # 将原始图像叠加在特征图上  
        
        # 去掉标题和坐标轴  
        plt.axis('off')  

    # 保存图像，去掉多余的边框  
    plt.savefig(imhs, bbox_inches='tight',dpi=100)  
    plt.show()  # 显示图像

class SimVP_Model(nn.Module):
    """SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

 
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=8, N_T=8, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = True
        self.enc1 = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc2 = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, 3, N_S, spatio_kernel_dec, act_inplace=act_inplace)
  
        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(hid_S*T, hid_T, N_T)
        else:
            self.hid = MidMetaNet(hid_S*T, hid_T, N_T,
            #self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        
        if model_type == 'incepu':
            self.hid1 = MidIncepNet(hid_S*T, hid_S*4, N_T)
        else:
            self.hid1 = MidMetaNet(hid_S*T, hid_S*4, N_T,
            #self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        

    def forward(self, x_raw1,x_raw2, **kwargs):
 ############This section is dedicated to extracting features of the "VV" part. begin#####################
        B, T, C, H, W = x_raw1.shape
      
        x1 = x_raw1.reshape(T* B,C, H, W)
       
        embed1, skip1 = self.enc1(x1,B)
        
        _, C_, H_, W_ = embed1.shape
        
        z1 = embed1.reshape(B, T, C_, H_, W_)
        ############This section is dedicated to extracting features of the "VV" part. end#####################
        ############This section is dedicated to extracting features of the "VV" part. begin#####################
        B, T, C, H, W = x_raw2.shape
        x2 = x_raw2.reshape(B*T, C, H, W)

        embed2, skip2 = self.enc2(x2,B)
        _, C_, H_, W_ = embed2.shape

        z2 = embed2.reshape(B, T, C_, H_, W_)
         #ooooo
        ############This section is dedicated to extracting features of the "VV" part. end#####################
        z = (z1+z2)/2
        skip = []
        #print(len(skip1))
        #show(picShow,"InAfter","/data/pth/liangjiaxuan-pth/result366In4After.png")
        for i in  range(4):
            skip.append((skip1[i]+skip2[i])/2)
        hid = self.hid(z)
        BS1,C1,H1,W1 = skip[3].shape
      #  print(skip[3].shape)

        hid1 = skip[3].reshape(B,13,C1,H1,W1)
        skip[3] = self.hid1(hid1).reshape(BS1,C1,H1,W1 )
        hid = hid.reshape(B*T, C_, H_, W_)
       
        Y = self.dec(hid,B,skip)
       # (Y.shape)
        Y = Y.reshape(B, T, 4, H, W)
        """
        # Step 2: 初始化一个用于存储每个时间步的 Importance Scores 的张量  
        importance_scores = torch.zeros(B, T)  # 形状为 (B, T)  

        # Step 3: 逐个时间步计算  
        for t in range(T):  
            # 提取当前时间步的第三个通道的数据  
            M = Y[:, t, 2, :, :]  # 形状为 (B, H, W)  

            # 创建布尔掩码，选择正值  
            positive_mask = M > 0  # 生成布尔掩码，True 表示正值  

            # 选择正值  
            M_positive = M[positive_mask]  # 选择正值  

            # 输出以验证正值的形状  
           

            # 计算正值的均值  
            if M_positive.numel() > 0:   
                # 计算均值  
                importance_scores[:, t] = M_positive.mean(dim=0)  # 将均值存储到对应的时间步  
            else:  
                importance_scores[:, t] = 0  # 如果没有正值，保持为0  

        # 输出每个时间步的 Importance Scores  
       # print("每个时间步的 Importance Scores:\n", importance_scores)  
       # print(importance_scores)
        globalss.modify_global_score(importance_scores)
        """
        A = Y[:, -1, :, :, :]
        
        return A


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvMYP(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvMYP(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x,bs):  # B*4, 3, 128, 128
      #  print(x.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        x = x.to(device)
        ex = []
        enc1 = self.enc[0](x,bs)
        latent = enc1
        ex.append(enc1)
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent,bs)
            ex.append(latent)
        return latent, ex


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvMYP(2*C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvMYP(2*C_hid, 4, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
    

    def forward(self, hid, bs,enc1=None):
        for i in range(0, len(self.dec)):
         #   print(1,hid.shape,enc1[3-i].shape)
            hid = torch.cat([hid, enc1[3-i]], dim=1)

            hid = self.dec[i](hid,bs)
        Y = hid
       
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
       
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[len(skips)-1-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        
       
        self.block = SimTASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)

        # self.block = MogaSubBlock(
        #         in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
       # B,  C, H, W = x.shape
       # print(x.shape)
        x = x.reshape(B, C*T, H, W)

        z = x
        
        for i in range(self.N2):
          #  print(i,z.shape)
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        y = z
        return y
    
            
     
class LSTMUNet(nn.Module):
    def __init__(self, sequence_length, in_channels, img_height, img_width, n_label, lstm_hidden_size=64):
        super(LSTMUNet, self).__init__()
        input_size = 3 * 256 * 256
        lstm_hidden_size = 512
        # LSTM层
        #config = ModelConfig()\
        # 假设的输入形状  
        input_shape = (13, 3, 64, 64)  # (batch_size, channels, height, width)  
  
        # 创建模型实例  
        self.lstm  = SimVP_Model(input_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA')  
        self.FinalConv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1)

        
        # U-Net架构
        
    def forward(self, x):
       
        BS =  x.shape[0]  
      #  print(x.shape)
        stacked_tensor1 = torch.empty(13,BS,3, 256, 256)
        stacked_tensor2 = torch.empty(13,BS,3, 256, 256)
      ##  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
      #  stacked_tensor.to(device)
        for i in range(13):
             n1 = x[:, 2*i:2*i+1, :, :, :]
             n2 = x[:, 2*i+1:2*i+2, :, :, :]
           #  print(n1.shape)
           #  n1 = n2/2+n1/2
             n1 = n1.reshape(BS,3,256,256)
             n2 = n2.reshape(BS,3,256,256)
            
             stacked_tensor1[i] = n1###VV
             stacked_tensor2[i] = n2###VH
        
        stacked_tensor1 = stacked_tensor1.permute(1,0,2,3,4)
        stacked_tensor2 = stacked_tensor2.permute(1,0,2,3,4)
     #   print(stacked_tensor.shape)
        lstm_out = self.lstm(stacked_tensor1,stacked_tensor2)
  
      #  show(lstm,save_path="a")
        #show(lstm,"InAfter","/data/pth/liangjiaxuan-pth/result366EndAfter.png")
     #  print(lstm.shape)
       
        return     lstm_out      
