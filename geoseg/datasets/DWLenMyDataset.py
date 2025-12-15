import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *
import warnings

"""CLASSES = ('Rice','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
           'A1','B1','C1','D1','E1','F1','G1','H1','I1','J1','K1','L1','M1','N1','O1','P1','Q1','R1','S1','T1','U1','V1','W1','X1','Y1','Z1',
           'A2','B2','C2','D2','E2','F2','G2','H2','I2','J2','K2','L2','M2','N2','O2','P2','Q2','R2','S2','T2','U2','V2','W2','X2','Y2','Z2',
           'A3','B3','C3','D3','E3','F3','G3','H3','I3','J3','K3','L3','M3','N3','O3','P3','Q3','R3','S3','T3','U3','V3','W3','X3','Y3','Z3',
           'A4','B4','C4','D4','E4','F4','G4','H4','I4','J4','K4','L4','M4','N4','O4','P4','Q4','R4','S4','T4','U4','V4','W4','X4','Y4','Z4')
        #   'A1','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee','Other','Accc','Bee','Cee','Rice','Other','Accc','Bee','Cee')

#CLASSES = ('Rice','Other','Accc','')
PALETTE = [[255,255,255],[127.5,0,0],[0,0,0],[127.5,0,127.5],[127.5,127.5,127.5],[127.5,255,127.5],[127.5,0,255],[127.5,127.5,255],[127.5,255,255],[0,0,0],[0,127.5,0],[0,255,0],[0,0,127.5],[0,127.5,127.5],[0,255,127.5],[0,0,255],[0,127.5,255],[0,255,255],[255,0,0],[255,127.5,0],[255,255,0],[255,0,127.5],[255,127.5,127.5],[255,255,127.5],[255,0,255],[255,127.5,255],[255,255,255]]
"""
CLASSES = ('Rice','Other','Accc','')
PALETTE = [[255,255,255],[127.5,0,0],[0,0,0],[127.5,0,127.5]]          

ORIGIN_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (256, 256)
TEST_IMG_SIZE = (256, 256)


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # crop_aug = SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=255, nopad=False)
    # img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
  #W  print("mg.shape, mask.shape",img.shape, mask.shape)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class LenMyDataset(Dataset):
    def __init__(self, data_root='/T2007061/liangjiaxvan_workspace/code/TESTData', mode='val', img_dir='TrainingImage', mask_dir='TrainingLabel',
                 img_suffix='.png', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
       
        
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
    def __getitem__(self, index):
        stacked_tensor = torch.empty(26,3, 256, 256)
        
        for i in range(26):
            p_ratio = random.random()
            if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
                img, mask = self.load_img_and_mask(index,i)
                if self.transform:
                    img, mask = self.transform(img, mask)
                else:
                    img, mask = np.array(img), np.array(mask)
            else:
              
                img, mask = self.load_mosaic_img_and_mask(index,i)
                if self.transform:
                    img, mask = self.transform(img, mask)
                else:
                    img, mask = np.array(img), np.array(mask)
                   
           
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            
            mask = torch.from_numpy(mask).long() #- 1
            img_id = self.img_ids[index]
           # print(img)
            stacked_tensor[i] = img
     #   print(stacked_tensor)
        results = {'img': stacked_tensor, 'gt_semantic_seg': mask, 'img_id': img_id}
      #  print(mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        a = len(mask_filename_list)
        print(a)
        print(len(img_filename_list))
        
        img_ids = [str(id.split('_')[0]) for id in mask_filename_list]
        return img_ids
    """  
            加载地理数据集图像文件  
        
            参数：  
                path (str): 图像文件的路径  
                grayscale (bool, 可选): 是否为灰度图，默认为False  
        
            返回：  
                src_img (np.ndarray): 加载的图像数据，数据类型为np.ndarray  
    """  

    def load_img_and_mask(self, index,i=0):
            
           
            img_ids = self.img_ids[index]
            StringId  = str(i)
            img_name = osp.join(self.data_root, self.img_dir,StringId,img_ids +'_B'+StringId+ self.img_suffix)
            my_int = int(img_ids)
           # print(img_ids +'_B'+StringId+ self.img_suffix)
            mask_name = osp.join(self.data_root, self.mask_dir, img_ids +'_M0'+ self.mask_suffix)
           # print(img_id,img_name, mask_name)
            
            img = Image.open(img_name).convert('RGB')
            mask = Image.open(mask_name).convert('L')
            return img, mask
    def load_mosaic_img_and_mask(self, index,i):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
      #  print(indexes)
        img_a, mask_a = self.load_img_and_mask(indexes[0],i)
        img_b, mask_b = self.load_img_and_mask(indexes[0],i)
        img_c, mask_c = self.load_img_and_mask(indexes[0],i)
        img_d, mask_d = self.load_img_and_mask(indexes[0],i)
       
        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # print(img.shape)
     
        return img, mask





def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
 
    seg_id = seg_list
    img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
    img_seg = img_seg.astype(np.uint8)
    img_seg = Image.fromarray(img_seg).convert('P')
    img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
    img_seg = np.array(img_seg.convert('RGB'))
    mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    img_id = str(seg_id.split('.')[0])+'.tif'
    img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[ 0].set_axis_off()
    ax[ 0].imshow(img)
    ax[ 0].set_title('RS IMAGE ' + img_id)
    ax[ 1].set_axis_off()
    ax[ 1].imshow(mask)
    ax[1].set_title('Mask True ' + seg_id)
    ax[ 2].set_axis_off()
    ax[ 2].imshow(img_seg)
    ax[ 2].set_title('Mask Predict ' + seg_id)
    ax[ 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    print("pp",patches)
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[ 0].set_axis_off()
        ax[ 0].imshow(img)
        ax[ 0].set_title('RS IMAGE '+img_id)
        ax[1].set_axis_off()
        ax[1].imshow(img_seg)
        ax[1].set_title('Seg IMAGE '+seg_id)
        ax[ 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.png')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

