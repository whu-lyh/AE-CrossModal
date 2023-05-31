import os
import numpy as np
import imageio
from tqdm import *

'''
The reason to calculate the mean and std value of custom dataset is here:
https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
The mean and std for raw equalrectangular images.
'''

 # the path of dataset
dirpath = "/root/public/data/Kitti/kitti360/data_2d_pano/"
seqs = [0, 2, 3, 4, 6, 7, 9, 10]

R_channel = 0
G_channel = 0
B_channel = 0
img_num = 0
for seq in seqs:
    seq_name = "2013_05_28_drive_%04d_sync"%seq
    filepath = os.path.join(dirpath, seq_name, "pano", "data_rgb")
    pathDir = os.listdir(filepath)
    for idx in tqdm(range(len(pathDir))):
        filename = pathDir[idx]
        img = imageio.imread(os.path.join(filepath, filename))
        img = img/255.0
        img_num = img_num + 1
        R_channel = R_channel + np.sum(img[:,:,0])
        G_channel = G_channel + np.sum(img[:,:,1])
        B_channel = B_channel + np.sum(img[:,:,2])
    print(seq_name + "Done!")

img_size  = img_num * 1400 * 2800
R_mean = R_channel / img_size
G_mean = G_channel / img_size
B_mean = B_channel / img_size
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))

R_channel = 0
G_channel = 0
B_channel = 0
for seq in seqs:
    seq_name = "2013_05_28_drive_%04d_sync"%seq
    filepath = os.path.join(dirpath, seq_name, "pano", "data_rgb")
    pathDir = os.listdir(filepath)
    for idx in tqdm(range(len(pathDir))):
        filename = pathDir[idx]
        img = imageio.imread(os.path.join(filepath, filename))
        img = img/255.0
        R_channel = R_channel + np.sum((img[:,:,0] - R_mean)**2)
        G_channel = G_channel + np.sum((img[:,:,1] - G_mean)**2)
        B_channel = B_channel + np.sum((img[:,:,2] - B_mean)**2)

R_var = (R_channel / img_size)**0.5
G_var = (G_channel / img_size)**0.5
B_var = (B_channel / img_size)**0.5
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))