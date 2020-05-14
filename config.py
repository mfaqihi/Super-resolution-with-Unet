from easydict import EasyDict as edict
import numpy as np

config = edict()

config.network = edict()
config.network.dims = 1024
config.network.n_channels = 13
config.network.out_channels = 2
config.network.loss = 'mean_squared_error'

config.train = edict()
config.train.lr = 1e-4
config.train.epochs = 1000
config.train.batch_size = 16
config.train.study = '13channels'
config.train.patience = 10
config.train.pretrained_weights = 'my_model_weights.h5'

config.crop = edict()
config.crop.width = 1024
config.crop.height = 1024
config.crop.S = 5
config.crop.merge_size = int(config.network.dims - 2 * config.crop.S)
config.crop.v_num = int(np.ceil(config.crop.height/config.crop.merge_size))
config.crop.h_num = int(np.ceil(config.crop.width/config.crop.merge_size))

config.test = edict()
config.test.batch_size = 9
config.test.raw_max = 4095
config.test.raw_min = 275
config.test.I_max = 496.63861083984375
config.test.I_min = 209.36285400390625
config.test.P_max = 0.42244160175323486
config.test.P_min = -0.46443378925323486