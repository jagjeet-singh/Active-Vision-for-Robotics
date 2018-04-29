from torch.utils.data import Dataset
import numpy as np
import pdb
import torch
import os
from PIL import Image
import pdb
class AV(Dataset):

    def __init__(self, img_list, mask_list, target_list, image_folder, mask_folder, target_folder, mode='train', transform=None):
        self.mode = mode
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_folder = target_folder
        self.transform = transform
        self.image_names, self.mask_names, self.target_names = img_list, mask_list, target_list
        if len(self.image_names) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode != 'test':
            img = Image.open(self.image_folder+'/'+self.image_names[idx]).convert('RGB')
            mask = Image.open(self.mask_folder+'/'+self.mask_names[idx]).convert('L')
            target = Image.open(self.target_folder+'/'+self.target_names[idx]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                mask = self.transform(mask).squeeze(0)
                target = self.transform(target)
            return img, target, mask
        else:
            img = Image.open(self.image_folder+self.image_names[idx]).convert('RGB')
            target = Image.open(self.target_folder+'/'+self.target_names[idx]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                target = self.transform(target)
            return img, target