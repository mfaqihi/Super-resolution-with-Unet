import numpy as np 
import os
from skimage.external import tifffile
from skimage import exposure
from config import config
from imgtolist import search

# source data structure:
# --inputs
#   --img1
#       --raw.tif
#   --img2
#       --raw.tif
#   --...
# --target
#   --img1.FPM
#       --XXXXXX
#           --..I.tif
#           --..P.tif
#   --img2.FPM
#       --XXXXXX
#           --..I.tif
#           --..P.tif
#   --...

# crop data structure:
# --raw
#   --img1
#       --0_0.tif (13 channels)
#       --...
#   --img2
#       --...
# --label
#   --img1
#       --0_0.tif (2 channels)
#       --...
#   --img2
#       --...


def crop_raw(input_size=config.network.dims, S=config.crop.S,
            v_num=config.crop.v_num, h_num=config.crop.h_num,
            width=config.crop.width, height=config.crop.height):

    mainpath = './TRIED_in_out/'
    searchpath=search('raw.tif',mainpath)
    for filepath in searchpath:
        # read full image (12 bits)
        raw = tifffile.imread(filepath)
        # center img
        raw_center = raw[:,515:1540,716:1741]
        tifffile.imsave(filepath.replace('raw','raw_center'),raw_center)

        # store 16 bits full image for better visualization
        raw_16 = exposure.rescale_intensity(raw_center, in_range=(0, 2**12 - 1))
        tifffile.imsave(filepath.replace('raw','raw_16'),raw_16)

        # full image padding
        merge_size = input_size - 2*S

        pad_size_x = v_num*merge_size + S - height
        pad_size_y = h_num*merge_size + S - width
        img_pad = np.pad(raw_center,((0,0),(S,pad_size_x),(S,pad_size_y)),'reflect')

        # crop
        output_path = '/'.join(filepath.replace('TRIED_in_out','data').split('\\')[:-1])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i in range(v_num):
            for j in range(h_num):
                cropped = img_pad[:,i*merge_size:i*merge_size+input_size, j*merge_size:j*merge_size+input_size]
                # named by the left upper point coordinate "y_x.tif"
                filename = output_path+'/'+str(i*merge_size)+'_'+str(j*merge_size)+'.tif'
                tifffile.imsave(filename,cropped)
        print('crop completed for raw: ',filepath.split('\\')[-2])


def crop_target(input_size=config.network.dims*2, S=config.crop.S*2,
            v_num=config.crop.v_num, h_num = config.crop.h_num,
            width = config.crop.width*2, height = config.crop.height*2):

    mainpath = './TRIED_in_out/'
    searchpath=search('I.tif',mainpath)
    for filepath in searchpath:
        # read full image (32 bits)
        raw_I = tifffile.imread(filepath)
        raw_P = tifffile.imread(filepath.replace('I.tif','P.tif'))

        # full image padding
        merge_size = input_size - 2*S
        pad_size_x = v_num*merge_size + S - height
        pad_size_y = h_num*merge_size + S - width

        img_I_pad = np.pad(raw_I,((S,pad_size_x),(S,pad_size_y)),'reflect')
        img_P_pad = np.pad(raw_P,((S,pad_size_x),(S,pad_size_y)),'reflect')

        # crop
        output_path = '\\'.join(filepath.replace('TRIED_in_out','data').split('\\')[:-3])+'/'+filepath.split('\\')[-3].split('.')[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i in range(v_num):
            for j in range(h_num):
                cropped_I = img_I_pad[i*merge_size:i*merge_size+input_size, j*merge_size:j*merge_size+input_size]
                cropped_P = img_P_pad[i*merge_size:i*merge_size+input_size, j*merge_size:j*merge_size+input_size]
                # named by the left upper point coordinate "y_x.tif"
                cropped = np.empty((2,input_size,input_size)).astype('float32')
                cropped[0,:,:] = cropped_I
                cropped[1,:,:] = cropped_P
                # named by the left upper point coordinate "y_x.tif"
                filename = output_path+'/'+str(int(i*merge_size/2))+'_'+str(int(j*merge_size/2))+'.tif'
                tifffile.imsave(filename,cropped)
        print('crop completed for target: ',filepath.split('\\')[-3].split('.')[0])


# crop_raw()
#crop_target()