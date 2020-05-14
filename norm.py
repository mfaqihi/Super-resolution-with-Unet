import os
import numpy as np
from skimage.external import tifffile


def search(s,mainpath):
    rootdir = mainpath
    searchpath = np.zeros(0)
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.find(s) != -1:# and filename.find(t) != -1:
                searchpath = np.append(searchpath, 
                        os.path.abspath(os.path.join(parent,filename)))
    return searchpath

mainpath = '../TRIED_dataset/'
serchpath=search('raw_center.tif',mainpath)

max_value = []
min_value = []

for im_list in serchpath:
    img = tifffile.imread(im_list)
    max_value.append(np.max(img))
    min_value.append(np.min(img))


print('max value: ',np.max(max_value),' min value: ',np.min(min_value))