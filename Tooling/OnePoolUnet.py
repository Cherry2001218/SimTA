import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolattnformer_s': _cfg(crop_pct=0.9),
    'poolattnformer_m': _cfg(crop_pct=0.95),
}


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
       
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


# class Pooling(nn.Module):
#     """
#     Implementation of pooling for PoolFormer
#     --pool_size: pooling size
#     """
#     def __init__(self, pool_size=3):
#         super().__init__()
#         self.pool = nn.AvgPool2d(
#             pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
#
#     def forward(self, x):
#         return self.pool(x) - x

class PoolAttn(nn.Module):
    """
    Implementation of pooling for PoolAttnFormer
    --pool_size: pooling size
    """
    def __init__(self, dim=256, norm_layer=GroupNorm):
        super().__init__()
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.patch_pool2 = nn.AdaptiveAvgPool2d((4, None))

        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((4, None))

        # self.act = act_layer()
        self.norm = norm_layer(dim)
        # self.proj = nn.Conv2d(dim,dim,1)
        self.proj0 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B,C,H,W = x.shape
        x_patch_attn1 = self.patch_pool1(x)
        x_patch_attn2 = self.patch_pool2(x)
        x_patch_attn = x_patch_attn1 @ x_patch_attn2
        x_patch_attn = self.proj0(x_patch_attn)

        x1 = x.view(B,C,H*W).transpose(1,2).view(B,H*W, 32, -1)
        x_embdim_attn1 = self.embdim_pool1(x1)
        x_embdim_attn2 = self.embdim_pool2(x1)
        x_embdim_attn = x_embdim_attn1 @ x_embdim_attn2

        x_embdim_attn = x_embdim_attn.view(B, H * W, C).transpose(1, 2).view(B, C, H, W)
        x_embdim_attn = self.proj1(x_embdim_attn)

        x_out  = self.norm(x_patch_attn + x_embdim_attn)
        x_out = self.proj2(x_out)
        return x_out

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.token_mixer = PoolAttn(dim=dim, norm_layer=norm_layer)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolAttnFormer.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks
###################Decoder#######################################

class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2*out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        ins = in_channels*2
      #  print(int(ins))
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2*out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
      
       # print(x1.shape)
       # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
       # print(x.shape)
       # print("aaaaaaaaaaaaaaaaaaaaa")        
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UpNormalT(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2,x3):
        x1 = self.up(x1)
        x2 = x3+x2
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

import torch.nn.functional as F
import matplotlib.pyplot as plt  
class DOECODER(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(DOECODER, self).__init__()
       # self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.bilinear = bilinear

        self.inc = Down(3, 32)
        self.downPre = Down(32,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 =Down(512,1024)

        self.BN1 = nn.BatchNorm2d(512)
        self.BN2 = nn.BatchNorm2d(256)
        self.BN3 = nn.BatchNorm2d(128)
        self.BN4 = nn.BatchNorm2d(64)
        self.up1 = UpNormalT(1024,512,bilinear)
        self.up2 = UpNormalT(512, 256, bilinear)
        self.up3 = UpNormalT(256, 128, bilinear)
        self.up4 = UpNormalT(128, 64, bilinear)
        self.outc = Up(64, 32,bilinear)
        self.output = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(32, 3, kernel_size=1))
    def forward(self,m, x):
       # print("(1)",x[6].shape)
        
        x0 = self.inc(m)
        x1 = self.downPre(x0)
       
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x[6] =  self.BN1(x[6])
        n = self.up1(x5, x4,x[6])
        x[4] =  self.BN2(x[4])
        n = self.up2(n,x3,x[4])
   
        x[2] =  self.BN3(x[2])
        n = self.up3(n,x2,x[2])
   
        x[0] =  self.BN4(x[0])
        n = self.up4(n,x[0],x1)
       
      #  print("up4",n.shape)
        n = self.outc(n,x0)
      #  print("Up5",n.shape)
      
        n = self.output(n)
      #  print("End",n.shape)
        
       
        
        return n




####################Decoder End################################
#####################TransNext#####################################



class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        # Generate sequnce length scale
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(input_resolution[0] * input_resolution[1])),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        # Use MLP to generate continuous relative positional bias
        rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                   relative_pos_index.view(-1)].view(-1, N, N)

        # Calculate attention map using sequence length scaled cosine attention and query embedding
        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * self.seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockX(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                fixed_pool_size=fixed_pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x
####################TransNext###################################

class PoolAttnFormer(nn.Module):
  
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 pool_size=3,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):

        super().__init__()
        self.decoder = DOECODER(n_classes =3)
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = BlockX(embed_dims[i], i, layers,
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

    
    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        xList = []
        for idx, block in enumerate(self.network):
          
            x = block(x)
           
  
            xList.append(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                
                outs.append(x_out)
            
    
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return xList

    def forward(self, x):
        m = x
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        xList = self.forward_tokens(x)
      
        
        x = self.decoder(m=m,x=xList)
      #  print(x)
       
        
        return x


@register_model
def poolattnformer_s12(pretrained=False, **kwargs):
    """
    poolattnformer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolAttnFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolattnformer_s']
    # if pretrained:
    #     url = model_urls['poolattnformer_s12']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    #     model.load_state_dict(checkpoint)
    return model

@register_model
def poolattnformer_s24(pretrained=False, **kwargs):
    """
    poolattnformer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolAttnFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolattnformer_s']
    # if pretrained:
    #     url = model_urls['poolattnformer_s24']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    #     model.load_state_dict(checkpoint)
    return model
#################################


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        in_channels: int 输入特征图的通道数
        out_channels: int 输出特征图的通道数
        kernel_size: (int, int) 卷积核的宽和高
        bias: bool 是否使用偏置
        """
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # 需要强制进行padding以保证每次卷积后形状不发生变化
        # 根据之前第4.3.2节内容的介绍，在stride=1的情况下，padding = kernel_size // 2
        # 如：卷积核为3×3则需要padding=1即可
        # 在下面的卷积操作中stride使用的是默认值1
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.out_channels,
                              out_channels=4 * self.out_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, last_state):
        """

        :param input_tensor: 当前时刻的输入x_t, 形状为[batch_size, in_channels, height, width]
        :param last_state: 上一时刻的状态c_{t-1}和h_{t-1}, 形状均为 [batch_size, out_channels, height, width]
        :return:
        """
        h_last, c_last = last_state
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
        # 确保 input_tensor 和 h_last 都在同一个设备上  
        input_tensor = input_tensor.to(device)  
        h_last = h_last.to(device) 
        combined_input = torch.cat([input_tensor, h_last], dim=1)
        # [batch_size, in_channels+out_channels, height, width]

       # print(combined_input.shape)
        combined_conv = self.conv(combined_input)  # [batch_size, 4 * out_channels, height, width]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        # 分割得到每个门对应的卷积计算结果，形状均为 [batch_size, out_channels, height, width]
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_last + i * g  # [batch_size, out_channels, height, width]
        h_next = o * torch.tanh(c_next)  # [batch_size, out_channels, height, width]
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        初始化记忆单元的C和H
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.out_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.out_channels, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        in_channels: 输入特征图的通道数，为整型
        out_channels: 每一层输出特征图的通道数，可为整型也可以是列表；
                      为整型时表示每一层的输出通道数均相等，为列表时则列表的长度必须等于num_layer
                      例如 out_channels =[32,64,128] 表示3层ConvLSTM的输出特征图通道数分别为
                      32、64和128，且此时的num_layer也必须为3
        kernel_size:  每一层中卷积核的长和宽，可以为一个tuple，如(3,3)表示每一层的卷积核窗口大小均为3x3；
                      也可以是一个列表分别用来指定每一层卷积核的大小，如[(3,3),(5,5),(7,7)]表示3层卷积各种的窗口大小
                      此时需要注意的是，如果为列表也报保证其长度等于num_layer
        num_layers: ConvLSTM堆叠的层数
        batch_first: 输入数据的第1个维度是否为批大小
        bias: 卷积中是否使用偏置
        return_all_layers: 是否返回每一层各个时刻的输出结果

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
        [Batch_size, Time_step, Channels, Height, Width]  or [Time_step, Batch_size, Channels, Height, Width]
    Output:
        当return_all_layers 为 True 时：
        layer_output_list: 每一层的输出结果，包含有num_layer个元素的列表，
                           每个元素的形状为[batch_size, time_step, out_channels, height, width]
        last_states: 每一层最后一个时刻的输出结果，同样是包含有num_layer个元素的列表，
                     列表中的每个元素均为一个包含有两个张量的列表，
                     如last_states[-1][0]和last_states[-1][1]分别表示最后一层最后一个时刻的h和c
                     layer_output_list[-1][:, -1] == last_states[-1][0]
                     shape:  [Batch_size, Channels, Height, Width]

        当return_all_layers 为 False 时：
        layer_output_list: 最后一层每个时刻的输出，形状为 [batch_size, time_step, out_channels, height, width]
        last_states: 最后一层最后一个时刻的输出，形状为 [batch_size, out_channels, height, width]

    Example:
        >> model = ConvLSTM(in_channels=3,
                 out_channels=2,
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=True)
        x = torch.rand((1, 4, 3, 5, 5)) # [batch_size, time_step, channels, height, width]
        layer_output_list, last_states = model(x)
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # 检查kernel_size是否符合上面说的取值情况

        # Make sure that both `kernel_size` and `out_channels` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        out_channels = self._extend_for_multilayer(out_channels, num_layers)
        # 将kernel_size和out_channels扩展到多层时的情况

        if not len(kernel_size) == len(out_channels) == num_layers:
            raise ValueError('len(kernel_size) == len(out_channels) == num_layers 三者的值必须相等')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 实例化每一层的ConvLSTM记忆单
            cur_in_channels = self.in_channels if i == 0 else self.out_channels[i - 1]
            # 当前层的输入通道数，除了第一层为self.in_channels之外，其它的均为上一层的输出通道数

            cell_list.append(ConvLSTMCell(in_channels=cur_in_channels, out_channels=self.out_channels[i],
                                          kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        # 必须要放到nn.ModuleList，否则在GPU上云运行时会报错张量不在同一个设备上的问题

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor: [Batch_size, Time_step, Channels, Height, Width]  or
                        [Time_step, Batch_size, Channels, Height, Width]
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
   #     print(input_tensor.shape)
        
       # if not self.batch_first:
            # 将(t, b, c, h, w) 转为 (b, t, c, h, w)
        #    input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        batch_size, time_step, _, height, width = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=batch_size,
                                             image_size=(height, width))

        layer_output_list = []  # 保存每一层的输出h，每个元素的形状为[batch_size, time_step, out_channels, height, width]
        last_state_list = []  # 保存每一层最后一个时刻的输出h和c，即[(h,c),(h,c)...]
        cur_layer_input = input_tensor  # [batch_size, time_step, in_channels, height, width]
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # 开始遍历每一层的ConvLSTM记忆单元，并取对应的初始值
            # h 和 c 的形状均为[batch_size, out_channels, height, width]
            output_inner = []
            cur_layer_cell = self.cell_list[layer_idx]  # 为一个ConvLSTMCell记忆单元
            for t in range(time_step):  # 对于每一层的记忆单元，按照时间维度展开进行计算
                h, c = cur_layer_cell(input_tensor=cur_layer_input[:, t, :, :, :], last_state=[h, c])###################
                output_inner.append(h)  # 当前层，每个时刻的输出h, 形状为 [batch_size, out_channels, height, width]

            layer_output = torch.stack(output_inner, dim=1)  # [batch_size, time_step, out_channels, height, width]
            cur_layer_input = layer_output  # 当前层的输出h，作为下一层的输入
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        init_states中的每个元素为一个tuple，包含C和H两个部分，如 [(h,c),(h,c)...]
        形状均为 [batch_size, out_channels, height, width]
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):  # 初始化每一层的初始值
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMKTH(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.conv_lstm = ConvLSTM(config.in_channels, config.out_channels,
                                  config.kernel_size, config.num_layers, config.batch_first)
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2, padding=2)
        self.hidden_dim = (config.width * config.height) // 4 * self.conv_lstm.out_channels[-1]
        # 除以4是因为长宽均要除以stride, 使用self.conv_lstm.out_channels[-1]
        # 主要是为了兼容out_channels传入整型或列表的情况，因为传入整型的话在ConvLSTM的初始化方法中_extend_for_multilayer()
        # 方法也会将其扩充一个list
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(self.hidden_dim, config.num_classes))

    def forward(self, x, labels=None):
   
        _, layer_output = self.conv_lstm(x)
       # print(layer_output[-1][0].shape)
        return layer_output[-1][0]


class ModelConfig(object):
    def __init__(self):
        self.num_classes = 6
        self.in_channels = 3
        self.out_channels = [32, 16, 4]
        self.kernel_size = [(3, 3), (5, 5), (7, 7)]
        self.num_layers = len(self.out_channels)
        self.batch_size = 2
        self.height = 256
        self.width = 256
        self.batch_first = True
        self.time_step = 10
class ASFFBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASFFBasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ASFF(nn.Module):
    def __init__(self, in_channels):
        super(ASFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        w = self.sigmoid(y1 + y2)
        out = w * x1 + (1 - w) * x2
        return out

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.block1 = ASFFBasicBlock(3, 64)
        self.block2 = ASFFBasicBlock(64, 128)
        self.asff = ASFF(in_channels=128)
        self.output_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x1, x2):
        out1 = self.block1(x1)
        out1 = self.block2(out1)
        out2 = self.block1(x2)
        out2 = self.block2(out2)
        out = self.asff(out1, out2)
        out = self.output_conv(out)
        return out

def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True))
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, feats=8, pad_value=None, zero_pad=True):
        super(UNet3D, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.pad_value = pad_value
        self.zero_pad = zero_pad

        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)
        self.final = nn.Conv3d(feats * 2, n_classes, kernel_size=3, stride=1, padding=1)
        # self.fn = nn.Linear(timesteps, 1)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x, batch_positions=None):
        out = x.permute(0, 2, 1, 3, 4)
        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)  # BxT pad mask
            if self.zero_pad:
                out[out == self.pad_value] = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        out = out.to(device)
        en3 = self.en3(out)
        pool_3 = self.pool_3(en3)
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        center_in = self.center_in(pool_4)
        center_out = self.center_out(center_in)
        concat4 = torch.cat([center_out, en4[:, :, :center_out.shape[2], :, :]], dim=1)
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        concat3 = torch.cat([trans3, en3[:, :, :trans3.shape[2], :, :]], dim=1)
        dc3 = self.dc3(concat3)
        final = self.final(dc3)
        final = final.permute(0, 1, 3, 4, 2)  # BxCxHxWxT

        # shape_num = final.shape[0:4]
        # final = final.reshape(-1,final.shape[4])
        if self.pad_value is not None:
            if pad_mask.any():
                # masked mean
                pad_mask = pad_mask[:, :final.shape[-1]] #match new temporal length (due to pooling)
                pad_mask = ~pad_mask # 0 on padded values
                out = (final.permute(1, 2, 3, 0, 4) * pad_mask[None, None, None, :, :]).sum(dim=-1) / pad_mask.sum(
                    dim=-1)[None, None, None, :]
                out = out.permute(3, 0, 1, 2)
            else:
                out = final.mean(dim=-1)
        else:
            out = final.mean(dim=-1)
        # final = self.dropout(final)
        # final = self.fn(final)
        # final = final.reshape(shape_num)

        return out
          
class LSTMUNet(nn.Module):
    def __init__(self, sequence_length, in_channels, img_height, img_width, n_label, lstm_hidden_size=64):
        super(LSTMUNet, self).__init__()
        input_size = 3 * 256 * 256
        lstm_hidden_size = 512
        # LSTM层
        in_channel = 3
        n_classes = 4
        config = ModelConfig()
        self.lstm = UNet3D(in_channel, n_classes)
        self.ASFF = FusionModel(num_classes=3)
        # U-Net架构
        self.unet1 = poolattnformer_s24()
        self.unet2 = poolattnformer_s24()
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
            # unet1 = self.ASFF(n1,n2)
             unet_out1 = self.unet1(n1)
             unet_out2 = self.unet2(n2)
             unet_out = unet_out1/2+unet_out2/2
             #unet_out = (n1+n2)/2
             stacked_tensor[i] = unet_out
        stacked_tensor = stacked_tensor.permute(1,0,2,3,4)
      
        lstm_out = self.lstm(stacked_tensor)
  
        return     lstm_out      
