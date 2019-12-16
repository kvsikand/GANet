from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
import sys
import shutil
import os
import re
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.GANet_deep import GANet

#from dataloader.data import get_test_set
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, help="crop height", default=384)
parser.add_argument('--crop_width', type=int, help="crop width", default=1248)
parser.add_argument('--max_disp', type=int, help="max disp", default=192)
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, help='kitti 2015? Default=True', default=1)
parser.add_argument('--data_path', type=str, help="data root", default='../data_scene_flow/training/')
parser.add_argument('--test_list', type=str, help="test list", default='lists/single_test.list')
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
parser.add_argument('--multi_gpu', type=int, default=0, help="multi_gpu choice")
parser.add_argument('--noise', type=str, default='ref', help="type of noise to add. One of ['none', 'gaussian', 'homography', 'rt']")

opt = parser.parse_args()

print(opt)

cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#print('===> Loading datasets')
#test_set = get_test_set(opt.data_path, opt.test_list, [opt.crop_height, opt.crop_width], false, opt.kitti, opt.kitti2015)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

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


def readPFM(file): 
    with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width

def add_noise(img, height, width, rmeans=None, rstdevs=None, mod_savename=None):
    def save_image(noisy):
        if not rmeans or not rstdevs:
          return None
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

        skimage.io.imsave(mod_savename, (shown).astype('uint8'))

    if opt.noise == 'gaussian':
        print("Adding Gaussian Noise...")
        #gaussian noise
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # rand_noise = np.random.normal(0, 1e-2, img.shape[:2])
        r = np.random.normal(r, 1e-2)
        g = np.random.normal(g, 1e-2)
        b = np.random.normal(b, 1e-2)
        # r = r + rand_noise
        # g = g + rand_noise
        # b = b + rand_noise

        noisy = np.zeros(img.shape)
        noisy[:, :, 0] = r
        noisy[:, :, 1] = g
        noisy[:, :, 2] = b

        save_image(noisy)

    elif opt.noise == 'homography':
        print("Adding Homography Noise...")
        noise_matrix = np.eye(3, 3)
        noise_matrix = noise_matrix + np.random.normal(0, 1e-20, noise_matrix.shape)

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
    elif opt.noise == 'perspective':
        print("Adding Perspective Noise...")
        noise_matrix = np.eye(3, 3)
        noise_matrix[2][0] = 1e-7

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
    elif opt.noise == 'shift':
        SHIFT = 30
        r = np.concatenate([np.zeros((SHIFT,img.shape[1])), img[SHIFT:, :, 0]], axis=0)
        g = np.concatenate([np.zeros((SHIFT,img.shape[1])), img[SHIFT:, :, 1]], axis=0)
        b = np.concatenate([np.zeros((SHIFT,img.shape[1])), img[SHIFT:, :, 2]], axis=0)
    elif opt.noise == 'trans':
        print("Adding Translation Noise...")
        noise_matrix = np.eye(3, 3)
        # noise_matrix = noise_matrix + np.random.normal(0, 1e-5, noise_matrix.shape)
        noise_matrix[0][2] = np.random.normal(0, 1e-5)
        noise_matrix[1][2] = np.random.normal(0, 1e-5)

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
    elif opt.noise == 'rot':
        print("Adding Rotation Noise...")
        noise_matrix = np.eye(3, 3)
        rot_theta = np.random.normal(0, 1e-5)
        rot_cos = np.cos(rot_theta)
        rot_sin = np.sin(rot_theta)
        noise_matrix[0][0] = rot_cos
        noise_matrix[1][1] = rot_cos
        noise_matrix[0][1] = rot_sin
        noise_matrix[1][0] = -rot_sin

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
    else: # no noise
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

    return r, g, b

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    r, g, b = add_noise(temp_data[3: 6, :, :].transpose(1, 2, 0), crop_height, crop_width)
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
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)
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
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    return temp

   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    avg_error = 0
    avg_rate = 0
    for index in range(len(filelist)):
        current_file = filelist[index]
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0
        elif opt.kitti:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0
        else:
            leftname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 14] + 'right/' + current_file[len(current_file) - 9:len(current_file) - 1]
            dispname = opt.data_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            savename = opt.save_path + str(index) + '.png'
            disp, height, width = readPFM(dispname)
        
        prediction = test(leftname, rightname, savename)
        mask = np.logical_and(disp >= 0.001, disp <= opt.max_disp)

        error = np.mean(np.abs(prediction[mask] - disp[mask]))
        rate = np.sum(np.abs(prediction[mask] - disp[mask]) > opt.threshold) / np.sum(mask)        
        avg_error += error
        avg_rate += rate
        print("===> Frame {}: ".format(index) + current_file[0:len(current_file)-1] + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    avg_error = avg_error / len(filelist)
    avg_rate = avg_rate / len(filelist)
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(len(filelist), avg_error, avg_rate))