from model import unet
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import backend as keras
from generator import DataGenerator
from imgtolist import imgtolist
import pandas as pd
from config import config
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============= Hyper parameter ============== #
epochs = config.train.epochs
study = config.train.study
patience = config.train.patience
dims = config.network.dims
n_channels = config.network.n_channels
input_size = (dims,dims,n_channels)


# ============= Training ===================== #

# get train list and validation list of data
train_list, val_list = imgtolist()
print('images: ')
print('train:', len(train_list), 'val:', len(val_list))

# get generator for batch training
training_generator = DataGenerator(train_list)
validation_generator = DataGenerator(val_list)

# define model/ checkpoint/ early stop
model = unet(input_size)

output_path = './result/'+study
if not os.path.exists(output_path):
    os.makedirs(output_path)
model_checkpoint = ModelCheckpoint(output_path+'/unet.hdf5', monitor='loss',verbose=0, save_best_only=False)
early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()



# train
history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs=epochs,
#                             use_multiprocessing=True,
                            callbacks=[model_checkpoint,early_stop,time_callback],
#                             workers=6,
                            verbose=1)
times = time_callback.times
doc=history.history
doc['times']=times
# store history
pd.DataFrame.from_dict(doc).to_csv('result/'+study+'/history.csv',index=False)

