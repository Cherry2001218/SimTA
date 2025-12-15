import torch
from torch import nn
import globalss

from PIL import Image 
import matplotlib.pyplot as plt
from openstl.modules import (ConvSC,TAUSubBlock,GroupConv2d)
import os
def show(feature, save_path=None,num =None):  
    # 将每个特征图从 GPU 转换为 CPU 并转为 numpy 数组  
    feature_map_data_list = [feature[0, 2].detach().cpu().numpy() ]  
    
    print(feature.shape)  
    nAddss = globalss.print_global_variable()  
    addss = str(nAddss)  

    # 生成要保存的文件夹路径  
    save_dir = '/data/pth/liangjiaxuan-pth/FvalCam/VH/'  
    os.makedirs(save_dir, exist_ok=True)  
    imhs = os.path.join(save_dir, f"{addss}.png")  
    OImg = '/data/pth/liangjiaxuan-pth/20231231Datas/ELabel/' + addss + "_M0.png"  
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
 
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.hid1 =  CBAM(T*(C+1))


    def forward(self, x_raw, **kwargs):
        
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip,att = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        BS,AC,AH,AW = att.shape
        hidA = att.view(B,T*AC,AH,AW)
        hidA = self.hid1(hidA)
        
        hidA = hidA.view(BS,AC,AH,AW )
        hid = hid.reshape(B*T, C_, H_, W_)
        
        Y = self.dec(hidA,hid, skip)
        Y = Y.view(B, T, C, H, W)
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
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )


        
        ####第一层AFMA实现###########################
        self.conv_img=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7),padding=3),

            nn.Conv2d(64, 4, kernel_size=(3,3), padding=1)
        )

        self.conv_feamap1=nn.Sequential(
            nn.Conv2d(C_hid, 4, kernel_size=(1, 1), stride=1)
        )
        self.patch_size = 10
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self._attention_on_depth =1 
        self.resolution_trans=nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2*self.patch_size * self.patch_size, bias=False),
            nn.Linear(2*self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )
        self.resolution_trans=nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2*self.patch_size * self.patch_size, bias=False),
            nn.Linear(2*self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )
        
  ####第一层AFMA实现###########################
    def forward(self, x):  # B*4, 3, 128, 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        x = x.to(device)
        ini_img=self.conv_img(x)
        attentions=[]
        enc1 = self.enc[0](x)
        latent = enc1
      
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        ####第一层AFMA实现###########################
        
        feamap = self.conv_feamap1(latent) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)
        for i in range(feamap.size()[1]):

                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)
            
                att=torch.unsqueeze(att,1)

                attentions.append(att)
       
        attentions = torch.cat((attentions), dim=1)
        
         
        ####第一层AFMA实现###########################
       # print(latent.shape,enc1.shape)
        return latent, enc1,attentions


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
         ####第一层AFMA实现###########################
        self.patch_size = 10
        self.att_depth =2
        self.conv_feamap_size = nn.Conv2d(3,4 ,kernel_size=(2**self.att_depth, 2**self.att_depth),stride=(2**self.att_depth, 2**self.att_depth),bias=False)
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))


          ####第一层AFMA实现###########################
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, attentions,hid, enc1=None):
        
        
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)

        correction=[]
        self.fold_layer = torch.nn.Fold(output_size=(Y.size()[-2], Y.size()[-1]), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        argx_feamap = self.conv_feamap_size(Y) / (2 ** self.att_depth * 2 ** self.att_depth)
        
        for i in range(Y.size()[1]):
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001,dim=-1)
            
            att = torch.matmul(attentions[:,i:i + 1, :, :]/non_zeros, torch.unsqueeze(self.unfold(argx_feamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2))

            att=torch.squeeze(att, dim=1)

            att = self.fold_layer(att.transpose(-1, -2))
           
            correction.append(att)

        correction=torch.cat(correction, dim=1)
        Y = correction * Y + Y
       
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
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

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

        
       
        self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
     

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
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y
    
            
     
class LSTMUNet(nn.Module):
    def __init__(self, sequence_length, in_channels, img_height, img_width, n_label, lstm_hidden_size=64):
        super(LSTMUNet, self).__init__()
        input_size = 3 * 256 * 256
        lstm_hidden_size = 512
        # LSTM层
        #config = ModelConfig()\
        # 假设的输入形状  
        input_shape = (13, 3, 256, 256)  # (batch_size, channels, height, width)  
  
        # 创建模型实例  
        self.lstm  = SimVP_Model(input_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA')  
        self.FinalConv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1)

        
        # U-Net架构
        
    def forward(self, x):
       
        BS =  x.shape[0]  
      #  print(x.shape)
        stacked_tensor = torch.empty(13,BS,3, 256, 256)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
        stacked_tensor.to(device)
        for i in range(13):
             n1 = x[:, 2*i:2*i+1, :, :, :]
             n2 = x[:, 2*i+1:2*i+2, :, :, :]
           #  print(n1.shape)
           #  n1 = n2/2+n1/2
             n1 = n1.reshape(BS,3,256,256)
             n2 = n2.reshape(BS,3,256,256)
             
            
             #unet_out2 = self.unet2(n2)
            # unet_out = unet_out1/2+unet_out2/2
             unet_out = n2
             stacked_tensor[i] = unet_out
        stacked_tensor = stacked_tensor.permute(1,0,2,3,4)
      
        lstm_out = self.lstm(stacked_tensor)
        lstm = self.FinalConv(lstm_out)
     #   show(lstm,"InAfter","/data/pth/liangjiaxuan-pth/result366EndAfter.png")
        return     lstm      
