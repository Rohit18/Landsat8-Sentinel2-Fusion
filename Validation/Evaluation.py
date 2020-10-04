import os
import tifffile as tiff
import numpy as np
import slidingwindow as sw
from pathlib import Path
from matplotlib import pyplot as plt
import scipy
from PIL import Image
import skimage
import gdal
import cv2
import sewar
import csv


directory = os.getcwd() + '\\LN_SR3_GAN\\'
pathlist = Path(directory).iterdir()

ssim_list = []
ergas_list = []
mse_list = []
mssim_list = []
psnr_list = []
rmse_list = []
rmsesw_list = []
sam_list = []
scc_list = []
uqi_list = []
vifp_list = []
filename_list = []

for path in pathlist:

    #load predicted image    
    pred_img = np.array(scipy.ndimage.imread(str(path)))
    pred_img = pred_img[:,:,1]
    
    #load target image
    filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
    filepath = os.getcwd() + '\\SR3_256\\' + filename + '.jpg'
    tar_img = np.array(scipy.ndimage.imread(str(filepath)))

    #evals
    filename_list.append(filename)
    ergas_list.append(sewar.full_ref.ergas(pred_img, tar_img, 1, 8)) #A
    mse_list.append(sewar.full_ref.mse(pred_img, tar_img)) #B
    mssim_list.append(sewar.full_ref.msssim(pred_img, tar_img)) #C
    psnr_list.append(sewar.full_ref.psnr(pred_img, tar_img)) #D
    rmse_list.append(sewar.full_ref.rmse(pred_img, tar_img)) #E
    #rmsesw_list.append(sewar.full_ref.rmse_sw(pred_img, tar_img))
    sam_list.append(sewar.full_ref.sam(pred_img, tar_img)) #F
    scc_list.append(sewar.full_ref.scc(pred_img, tar_img)) #G
    ssim_list.append(sewar.full_ref.ssim(pred_img, tar_img)) #H
    uqi_list.append(sewar.full_ref.uqi(pred_img, tar_img)) #I
    vifp_list.append(sewar.full_ref.vifp(pred_img, tar_img)) #J

eval_list = zip(filename_list, ergas_list, mse_list, mssim_list, psnr_list, rmse_list, sam_list, scc_list, uqi_list, vifp_list) #rmsew_missing

with open('LN_SR3_GAN_Eval.csv', 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerows(eval_list)




