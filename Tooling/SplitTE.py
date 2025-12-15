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
import warnings
import shutil 


def get_img_ids(data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        a = len(mask_filename_list)
        print(a)
        print(len(img_filename_list))
        
        img_ids = [str(id.split('_')[0]) for id in mask_filename_list]
        return img_ids

def load_img_and_mask(self, index,i=0):
            
           
            img_ids = self.img_ids[index]
            StringId  = str(i)

            img_name = osp.join(self.data_root, self.img_dir,img_ids +'_B'+StringId+ self.img_suffix)
            #img_name = osp.join(self.data_root, self.img_dir,img_ids +'_B'+StringId+ self.img_suffix)
            my_int = int(img_ids)
           # print(img_ids +'_B'+StringId+ self.img_suffix)
           
         #   print(img_ids,img_name, mask_name)
            
           

def Split(data_dir,img_dir,mask_dir,mvImg,mvMask):
        
        img_ids = get_img_ids(data_dir,img_dir,mask_dir)
        for img_id in img_ids:
            
            nNum = int(img_id)
            if random.random() >= 0.5:  
            
                if nNum < 1766:
                    mask_name = osp.join(data_dir, mask_dir, img_id +'_M0.png')
                    print(mask_name)
                    dst_file = os.path.join(data_dir, mvMask)
                
                    shutil.move(mask_name, dst_file)
                    for i in range(26):
                        StringId  = str(i)
                        img_name = osp.join(data_dir, img_dir,img_id +'_B'+StringId+".png")
                        print(img_name)
                        mvImgs = os.path.join(data_dir, mvImg)
                        shutil.move(img_name, mvImgs)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            
filepath = '/data/pth/liangjiaxuan-pth/20231231Datas'
img_dir = 'EImage'
mask_dir='ELabel'
mvImg = 'TImage'
mvLabel = 'TrainingLabel'
Split(filepath,img_dir,mask_dir,mvImg,mvLabel)