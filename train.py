from Unet import Unet
from tensorflow.python.keras.optimizers import adadelta,Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils import to_categorical
from DataSet import DataSet
import numpy as np
import config
import math
def TrainGenerator(dataset,batch_size):
    while True:
        batch_images, batch_labels = dataset.get_train_dataBatch(batchsize=batch_size)
        #batch_labels=batch_labels[:,:,np.newaxis]
        yield batch_images,batch_labels
def ValGenerator(dataset,batch_size):
    while True:
        batch_images, batch_labels = dataset.get_val_dataBatch(batchsize=batch_size)
        #batch_labels=batch_labels[:,:,np.newaxis]
        yield batch_images,batch_labels
if __name__=='__main__':
    dataSet=DataSet()
    trainGenerator=TrainGenerator(dataSet,config.batch_size)
    valGenerator=ValGenerator(dataSet,config.batch_size)
    #Fcn32.fit_generator(trainGenerator,steps_per_epoch=math.ceil(config.train_num / config.batch_size),epochs=config.epochs,validation_data=valGenerator,validation_steps=8)
    unet=Unet().build_model()
    unet.fit_generator(trainGenerator,steps_per_epoch=math.ceil(config.train_num / config.batch_size),epochs=config.epochs,validation_data=valGenerator,validation_steps=8)
    unet.save_weights('unet.h5')