from torch.utils.data import Dataset
import numpy as np
import pdb
import torch
import os

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def make_dataset(mode, image_folder, mask_folder):
    if mode == 'train' or mode == 'val':
        image_names = os.listdir(image_folder)
        mask_names = os.listdir(mask_folder)
        return image_names, mask_names
    else:
        image_names = os.listdir(image_folder)
        return image_names, None


class ActiveVisionDataset(Dataset):

    def __init__(self, data_list, mode='train', transform=None, image_folder = '../../images', mask_folder = '../../seg'):
        self.transform = transform
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_names, self.mask_names = make_dataset(mode, self.image_folder, self.mask_folder)
        if len(self.image_paths) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.type != 'test':
            img = Image.open(self.image_folder+self.image_names[idx]).convert('RGB')
            mask = Image.open(self.image_folder+self.mask_names[idx])
            if self.train_transform is not None:
                img = self.transform(img)
                mask = self.transform(mask)
            return img, mask
        else:
            img = Image.open(self.image_folder+self.image_names[idx]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img