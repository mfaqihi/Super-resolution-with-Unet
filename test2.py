import numpy as np
from model import unet, psnr, ssim
from generator import DataGenerator
from imgtolist import imgtolist
from keras.models import load_model
from skimage.external import tifffile
from config import config
import pandas as pd
import os
from skimage.measure import compare_mse as compare_mse
from skimage.measure import compare_psnr as compare_psnr
from skimage.measure import compare_ssim as compare_ssim
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============= Hyper parameter ============== #
batch_size = config.test.batch_size
study = config.train.study
S = config.crop.S * 2
merge_size = config.crop.merge_size * 2
v_num=config.crop.v_num
h_num = config.crop.h_num
#case = config.test.target

# ============= Testing ============== #

def search(s,mainpath):
    rootdir = mainpath
    searchpath = np.zeros(0)
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.find(s) != -1:# and filename.find(t) != -1:
                searchpath = np.append(searchpath, 
                        os.path.abspath(os.path.join(parent,filename)))
    return searchpath

def merge_all():
    # load model
    model = load_model('result/'+study+'/unet.hdf5', custom_objects={'psnr': psnr, 'ssim': ssim})
    model.save_weights('my_model_weights.h5')
    model = unet((config.network.dims,config.network.dims,config.network.n_channels),pretrained='my_model_weights.h5')
    mainpath = './data/test/in/'
    imgs = search('raw_center.tif',mainpath)
    y=np.empty(len(imgs))
    for imgname in imgs:
        # predict
        raw = tifffile.imread(imgname)
        raw = (raw-config.test.raw_min)/(config.test.raw_max-config.test.raw_min)
        raw=raw[[1,3,5,7,9,11],:1024,:1024]
        raw = np.transpose(raw,(1,2,0))
        raw = raw[np.newaxis, :]
        y_pred = model.predict(raw)
        y_pred[:,:,:,0]=(y_pred[:,:,:,0]*(config.test.I_max-config.test.I_min)+config.test.I_min)
        y_pred[:,:,:,1]=(y_pred[:,:,:,1]*(config.test.P_max-config.test.P_min)+config.test.P_min)
        y_pred = np.transpose(y_pred[0],(2,0,1)).astype('float32')
        output = imgname.replace('raw_center','result')
        output_path = '/'.join(output.split('/')[:-1])
        #if not os.path.exists(output):
         #   os.makedirs(output)
        tifffile.imsave(output,y_pred)
        np.append(y,y_pred)
        return y
        

merge_all()