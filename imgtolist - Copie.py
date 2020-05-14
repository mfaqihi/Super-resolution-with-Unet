import os
import numpy as np
from config import config


# data structure:
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

# the format of list:
#       path_merge = path_raw+' '+path_target


# For searching the file which name contains specific string in folder and sub folder
def search(s,mainpath):
    rootdir = mainpath
    searchpath = np.zeros(0)
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.find(s) != -1:# and filename.find(t) != -1:
                searchpath = np.append(searchpath, 
                        os.path.abspath(os.path.join(parent,filename)))
    return searchpath

def imgtolist(train=True, img=None):
    # train
    if train==True:
        all_images = [] # all train image paths
        train_list = [] # train image paths list after train-validation split
        val_list = []   # val   image paths list after train-validation split

        mainpath = './data/in/'
        serchpath=search('.tif',mainpath)
        for i in serchpath:
            all_images.append(i+' '+i.replace('in','out'))

        # split training set and validation set
        train_len = int(len(all_images)*0.7)
        train_list = all_images[:train_len]
        val_list = all_images[train_len:]

        return train_list, val_list

    else:
    # test
    # without shuffling for merging
        size = config.crop.merge_size
        v_num = config.crop.v_num
        h_num = config.crop.h_num
        test_list = []
        main_path = './data/test/'
        
        # read image according to the filename (coordinate)
        for i in range(v_num):
            for j in range(h_num):
                path_raw = main_path+'IN/'+img+'/'+str(i*size)+'_'+str(j*size)+'.tif'
                path_target = main_path+'OUT/'+img+'/'+str(i*size)+'_'+str(j*size)+'.tif'
                test_list.append(path_raw+' '+path_target)
        return test_list
    
    
    
    