"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.LenMyDataset import *
from geoseg.models.concat.SimVp import LSTMUNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import torchvision.transforms as transforms
# training hparam

# 些训练的超参数，包括最大训练周期、忽略的标签索引、训练和验证的批量大小、学习率、权重衰减、骨干网络的学习率和权重衰减、类别数量和类别列表
max_epoch = 100
ignore_index = 255
train_batch_size = 1
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
#last-v26.ckpt
#unetformer-r18-1024-768crop-e40

#设置模型权重的名称和路径、日志名称、监控指标、保存模型的策略、检查验证集的频率、预训练模型的路径、GPU设置和是否从检查点恢复训练
weights_name = "SIMVPConcatData2"
weights_path = "/data/pth/liangjiaxuan-pth/modelweight/SIMVPConcatData{}".format(weights_name)
test_weights_name = "SIMVPConcatData2"
log_name = 'uavid/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 5
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#这里定义了序列长度、批量大小、通道数、图像高度和宽度、标签数量。
sequence_length = 26
batch_size = 2
channels = 3
img_height = 256
img_width = 256
n_label = 2

# 创建模型
#sequence_length, channels, img_height, img_width, 3"""
net = LSTMUNet(sequence_length, channels, img_height, img_width, 3)
#  define the network

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = False
transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为PyTorch张量
    ])
# define the dataloader
filepath = '/data/pth/liangjiaxuan-pth/20231231Datas'

# 定义训练数据集，包括数据根目录、图像目录、掩码目录、模式、马赛克比率、转换操作和图像大小
train_dataset = LenMyDataset(data_root=filepath, img_dir='TImage', mask_dir='TrainingLabel',
                             mode='train', mosaic_ratio=0.25, transform=val_aug, img_size=(256, 256))

# 验证数据集
val_dataset = LenMyDataset(data_root=filepath, img_dir='EImage', mask_dir='ELabel', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(256, 256))
# 测试数据集                           
test_dataset = LenMyDataset(data_root=filepath, img_dir='EImage', mask_dir='ELabel', mode='test',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(256, 256))
#定义了训练数据加载器，设置了批量大小、工作线程数、是否固定内存、是否打乱数据和是否丢弃最后一个不完整的批次
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=2,
                         num_workers=0,
                         pin_memory=True,
                          shuffle=True,
                          drop_last=True)
#定义了验证数据加载器，参数与训练数据加载器类似，但不丢弃最后一个不完整的批次                       
val_loader = DataLoader(dataset=val_dataset,
                       batch_size=2,
                       num_workers=0,
                      shuffle=True,
                        pin_memory=True,
                        drop_last=False)


#定义测试集加载器
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=2,
                        num_workers=0,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=False)


#定义了优化器和学习率调度器
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
