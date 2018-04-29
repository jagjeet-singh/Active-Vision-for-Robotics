import os
import argparse
import sys
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import random
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import pdb

cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


def get_mask(curr_image, target_image, net):
    # start = time.time()
    net.eval()
    transform = transforms.ToTensor()
    curr_image = transform(curr_image)
    target_image = transform(target_image)
    curr_image = curr_image.view(1,curr_image.shape[0],curr_image.shape[1],curr_image.shape[2])
    target_image = target_image.view(1,target_image.shape[0],target_image.shape[1],target_image.shape[2])

    image_input = Variable(curr_image.type(FloatTensor), volatile=True)
    target_input = Variable(target_image.type(FloatTensor), volatile=True)
    output = net(image_input, target_input)
    if use_cuda:
        mask_output = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy() 
    else:
        mask_output = output.data.max(1)[1].squeeze_(1).squeeze_(0).numpy()
    kernel = np.ones((4,4), np.uint8)
    mask_output_processed = cv2.erode(mask_output.astype(np.float32), kernel, iterations=1)
    # print("Time for eval 1 image: {}".format(time.time()-start))
    return mask_output_processed

# def main():
#     curr_image = Image.open('../../active-vision-storage/av_data/images/2000.png').convert('RGB')
#     target_image = Image.open('../../active-vision-storage/av_data/targets/2000.png').convert('RGB')
#     # curr_image = Image.open('../../objects/cur2.png').convert('RGB').resize((256,256))
#     # target_image = Image.open('../../objects/tar2.png').convert('RGB').resize((256,256))
#     # snapshot_path = '../../active-vision-storage/av_fcn8s/epoch_5_loss_653.77799_acc_0.99585_acc-cls_0.97513_mean-iu_0.96923_fwavacc_0.99175_lr_0.0000100000.pth'
#     snapshot_path = '../../active-vision-storage/ckpt/alexnet/epoch_7_loss_750.40502_acc_0.99551_acc-cls_0.98263_mean-iu_0.96637_fwavacc_0.99118_lr_0.0000100000.pth'
#     # net = FCN8s(num_classes=2, pretrained=False)
#     net = FCN8s_alex(num_classes=2, pretrained=False)
#     if use_cuda:
#         net.cuda()
#         net.load_state_dict(torch.load(snapshot_path))
#     else:
#         net.load_state_dict(torch.load(snapshot_path,map_location=lambda storage, loc: storage))

#     pred_mask = get_mask(curr_image, target_image,net)
#     plt.imshow(pred_mask)
#     plt.show()

# if __name__ == '__main__':
#     main()