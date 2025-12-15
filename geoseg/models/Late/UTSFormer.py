import torch
import torch.nn as nn
import copy

import numpy as np

import torch.nn.functional as F


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn
 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn
 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormLocal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        # print('before fn: ', x.shape)
        x = self.fn(x, **kwargs)
        # print('after fn: ', x.shape)
        return x


class Conv1x1Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
            )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TSViTcls(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(13, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x):
        xh =x
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = xh.shape
      
        
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt *2).to(torch.int64)
        xt = F.one_hot(xt, num_classes=13).to(torch.float32)
        xt = xt.reshape(-1, 13)
       
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(xh)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
       
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x

class TST1(nn.Module):
    def __init__(self):
        super().__init__()
        model_config = {
            'img_res': 256,
            'patch_size': 8,
            'num_classes': 4,
            'max_seq_len': 13,
            'dim': 512,
            'temporal_depth': 6,
            'spatial_depth': 6,
            'heads': 4,
            'dim_head': 32,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'pool': 'cls',
            'scale_dim': 4,
            'num_channels': 9
        }

        self.TF = TSViTcls(model_config)
        

    def forward(self, x):
        #print("Aaaaaaaaaaa")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        x = x.to(device)
       
        n1 = self.TF(x).to(device)
        return n1

class TST2(nn.Module):
    def __init__(self):
        super().__init__()
        model_config = {
            'img_res': 128,
            'patch_size': 8,
            'num_classes': 4,
            'max_seq_len': 13,
            'dim': 512,
            'temporal_depth': 6,
            'spatial_depth': 6,
            'heads': 4,
            'dim_head': 32,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'pool': 'cls',
            'scale_dim': 4,
            'num_channels': 17
        }

        self.TF = TSViTcls(model_config)
        

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        x = x.to(device)
       
        n1 = self.TF(x).to(device)
        
        return n1



class TST3(nn.Module):
    def __init__(self):
        super().__init__()
         
        model_config = {
            'img_res': 64,
            'patch_size': 8,
            'num_classes': 4,
            'max_seq_len': 13,
            'dim': 512,
            'temporal_depth': 6,
            'spatial_depth': 6,
            'heads': 4,
            'dim_head': 32,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'pool': 'cls',
            'scale_dim': 4,
            'num_channels': 33
        }

        self.TF = TSViTcls(model_config)
        

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        x = x.to(device)
       
        n1 = self.TF(x).to(device)
        
        return n1




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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        x = x.to(device)
        X=self.double_conv(x).to(device)
        return X

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
        
        self.conv = DoubleConv(in_channels+4, out_channels)

    def forward(self, x1, x2,x3):
       
        x1 = self.up(x1)#上采样
        #input is CHW
      
        x = torch.cat([x2, x1,x3], dim=1) #特征整合
        return self.conv(x)
class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #self.lstm = CNNLSTM()
        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.up2 = Up(64, 32, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)
        self.ts1 = TST1()
        self.ts2 = TST2()
        self.ts3 = TST3()



    def forward(self, x):
       
        BS =  x.shape[0]
        # x1 = self.inc(x)
        stacked_tensor1 = torch.empty(BS,13,8, 256, 256)
        n1 = x.reshape(BS*13,3,256,256)
        x1 = self.inc(n1)
        
        stacked_tensor1 = x1.reshape(BS,13,8,256,256)
        _,_,C1,W1,H1=stacked_tensor1.shape
       


        x1 = stacked_tensor1[:,-1,:,:,:]
        x1 =  torch.squeeze(x1, dim=1)
        #x2 = self.down1(x1)


#################2

        stacked_tensor2 = torch.empty(BS,13,16, 128, 128)
        n2 = stacked_tensor1.reshape(BS*13,8,256,256)
        x2 = self.down1(n2)
        
        stacked_tensor2 = x2.reshape(BS,13,16,128,128)
        _,_,C2,W2,H2=stacked_tensor2.shape
       


        x2 = stacked_tensor2[:,-1,:,:,:]
        x2 =  torch.squeeze(x2, dim=1)
        #x2 = self.down1(x1)



###################2
      

#################3

        stacked_tensor3 = torch.empty(BS,13,32, 64, 64)
        n3 = stacked_tensor2.reshape(BS*13,16,128,128)
        x3 = self.down2(n3)
        
        stacked_tensor3 = x3.reshape(BS,13,32,64,64)
        _,_,C3,W3,H3=stacked_tensor3.shape
       


        x3 = stacked_tensor3[:,-1,:,:,:]
        x3 =  torch.squeeze(x3, dim=1)
        #x2 = self.down1(x1)

###################2
#################2

        stacked_tensor4 = torch.empty(BS,13,64, 32, 32)
        n4 = stacked_tensor3.reshape(BS*13,32,64,64)
        x4 = self.down3(n4)
        
        stacked_tensor4 = x4.reshape(BS,13,64,32,32)
        _,_,C4,W4,H4=stacked_tensor4.shape
       

      
        x4 = stacked_tensor4[:,-1,:,:,:]
        x4 =  torch.squeeze(x4, dim=1)
        #x2 = self.down1(x1)

###################2

        
        xTs1 = self.ts1(stacked_tensor1)
        xTs2 = self.ts2(stacked_tensor2)
        xTs3 = self.ts3(stacked_tensor3)
       
        x = self.up2(x4, x3,xTs3)
        x = self.up3(x, x2,xTs2)
        x = self.up4(x, x1,xTs1)
        logits = self.outc(x)
        return logits
    
class LSTMUNet(nn.Module):
    def __init__(self, sequence_length=13, in_channels=3, img_heigh=256, img_width=256, n_label=4, lstm_hidden_size=64):
        super(LSTMUNet, self).__init__()
        input_size = 3 * 256 * 256
        lstm_hidden_size = 512
        # LSTM层
        #config = ModelConfig()
        self.lstm1 = Unet()
        self.lstm2 = Unet()
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
        lstm_out1 = self.lstm1(stacked_tensor2)
        lstm_out2 = self.lstm2(stacked_tensor1)
        lstm_out = (lstm_out1+lstm_out2)/2
        return     lstm_out   
    
