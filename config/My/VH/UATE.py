"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.LenMyDataset import *
from geoseg.models.VH.UATE import LSTMUNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import torchvision.transforms as transforms
# training hparam
max_epoch =40 
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
weights_name = "UATEVH"
weights_path = "/data/pth/liangjiaxuan-pth/model_weights/UATE/{}".format(weights_name)
test_weights_name = "UATEVH"
log_name = 'UATE/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 5
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None
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


train_dataset = LenMyDataset(data_root=filepath, img_dir='TImage', mask_dir='TrainingLabel',
                             mode='train', mosaic_ratio=0.25, transform=val_aug, img_size=(256, 256))

val_dataset = LenMyDataset(data_root=filepath, img_dir='EImage', mask_dir='ELabel', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(256, 256))
test_dataset = LenMyDataset(data_root=filepath, img_dir='ThImage', mask_dir='ThLabel', mode='test',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(256, 256))

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=2,
                         num_workers=0,
                         pin_memory=True,
                          shuffle=True,
                          drop_last=True)
val_loader = DataLoader(dataset=val_dataset,
                       batch_size=2,
                       num_workers=0,
                      shuffle=True,
                        pin_memory=True,
                        drop_last=False)

# define the dataloader




test_loader = DataLoader(dataset=test_dataset,
                        batch_size=2,
                        num_workers=0,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=False)


# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
