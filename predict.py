from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
#from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, help="crop height", default=240)
parser.add_argument('--crop_width', type=int, help="crop width", default=624)
parser.add_argument('--max_disp', type=int, help="max disp", default=192)
parser.add_argument('--resume', type=str, help="resume from saved model", default='./trained_models/kitti2015_final.pth')
parser.add_argument('--cuda', type=bool, help='use cuda?', default=True)
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, help='kitti 2015? Default=False', default=1)
parser.add_argument('--data_path', type=str, help="data root", default='../data_scene_flow/testing/')
parser.add_argument('--test_list', type=str, help="test list", default='lists/single_test.list')
parser.add_argument('--save_path', type=str, help="location to save result")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")
parser.add_argument('--noise', type=str, default='ref', help="type of noise to add. One of ['none', 'gaussian', 'homography', 'rt']")

opt = parser.parse_args()


print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")
    
cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)
#print('===> Loading datasets')


print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
       
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def add_noise(img, height, width, rmeans, rstdevs):
    def save_image(noisy, name="test_noise.png"):
        r = noisy[:, :, 0]
        g = noisy[:, :, 1]
        b = noisy[:, :, 2]
        r = r*rstdevs[0] + rmeans[0]
        g = g*rstdevs[1] + rmeans[1]
        b = b*rstdevs[2] + rmeans[2]
        shown = np.zeros(noisy.shape)
        shown[:, :, 0] =  r
        shown[:, :, 1] =  g
        shown[:, :, 2] =  b

        skimage.io.imsave(name, (shown).astype('uint8'))

    if opt.noise == 'gaussian':
        #gaussian noise
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        r = r + np.reshape(np.random.normal(0, 25, height * width), (height, width))
        g = g + np.reshape(np.random.normal(0, 25, height * width), (height, width))
        b = b + np.reshape(np.random.normal(0, 25, height * width), (height, width))
    elif opt.noise == 'homography':
        print("Adding Homography Noise...")
        noise_matrix = np.eye(3, 3)
        noise_matrix[2][0] = 5e-5

        print("NOISE", noise_matrix)
        
        noisy = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_coord = noise_matrix.dot(np.array([i, j, 1]))
                new_coord = new_coord / new_coord[2]
                if new_coord[0] < 0 or new_coord[1] < 0:
                    continue
                if new_coord[0] >= img.shape[0] or new_coord[1] >= img.shape[1]:
                    continue
                noisy[int(new_coord[0])][int(new_coord[1])] = img[i][j]

        r = noisy[:, :, 0]
        g = noisy[:, :, 1]
        b = noisy[:, :, 2]

        save_image(noisy)
    elif opt.noise == 'cvperspective':
        corner_noise = np.random.normal(0, NOISE_AMT * .5, (4, 2)).astype(np.float32)
        corner_noise = np.clip(corner_noise, -NOISE_AMT, NOISE_AMT)
        # corner_noise = np.array([[30, 60], [-50, 50], [-100, -50], [50, -50]], np.float32)
        persp_matrix = cv2.getPerspectiveTransform(
            np.array([[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]], np.float32),
            np.array([[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]], np.float32) + corner_noise,
        )
        persp_matrix = np.around(persp_matrix, 3)
        print("PERSP", persp_matrix)

        noisy = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_coord = persp_matrix.dot(np.array([i, j, 1]))
                if new_coord[0] < 0 or new_coord[1] < 0:
                    continue
                if new_coord[0] >= img.shape[0] or new_coord[1] >= img.shape[1]:
                    continue
                noisy[int(new_coord[0])][int(new_coord[1])] = img[i][j]

        r = noisy[:, :, 0]
        g = noisy[:, :, 1]
        b = noisy[:, :, 2]

        save_image(noisy)
    elif opt.noise == 'shift':
        SHIFT = 30
        r = np.concatenate([np.zeros((SHIFT,img.shape[1])), img[SHIFT:, :, 0]], axis=0)
        g = np.concatenate([np.zeros((SHIFT,img.shape[1])), img[SHIFT:, :, 1]], axis=0)
        b = np.concatenate([np.zeros((SHIFT,img.shape[1])), img[SHIFT:, :, 2]], axis=0)
    elif opt.noise == 'rt':
        # TODO
        print("still need to implement")
    else: # no noise
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

    return r, g, b

def test_transform(temp_data, crop_height, crop_width, rmeans, rstdevs):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    
    left = np.ones([1, 3, crop_height, crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')

    r, g, b = add_noise(temp_data[3: 6, :, :].transpose(1, 2, 0), crop_height, crop_width, rmeans, rstdevs)

    right[0, 0, :, :] = r
    right[0, 1, :, :] = g
    right[0, 2, :, :] = b
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    means = (np.mean(r[:]), np.mean(g[:]), np.mean(b[:]))
    stdevs = (np.std(r[:]), np.std(g[:]), np.std(b[:]))
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data, means, stdevs

def test(leftname, rightname, savename):
  #  count=0
    data, means, stdevs = load_data(leftname, rightname)
    input1, input2, height, width = test_transform(data, opt.crop_height, opt.crop_width, means, stdevs)

    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)
     
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]

    max_depth = np.max(temp)
    min_depth = np.min(temp)

    temp = (temp - min_depth) / (max_depth - min_depth)

    plt.imsave(savename, temp)
    # skimage.io.imsave(savename, (temp * 256).astype('uint8'))

   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    for index in range(len(filelist)):
        current_file = filelist[index]
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
        if opt.kitti:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]

        if not opt.save_path:
            opt.save_path = './result_' + opt.noise + '/'
            
        savename = opt.save_path + current_file[0: len(current_file) - 1]
        test(leftname, rightname, savename)

