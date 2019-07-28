import numpy as np
import math
from tensorflow.python.keras import backend as K
def get_dice(gt,output):
    # gt & output 's dim should be [n , 1]
    assert gt.shape[0]==output.shape[0]
    right=gt*output
    return np.sum(right)*2.0/(np.sum(gt)+np.sum(output))

def tp_sum(y_true,y_pred):
    y_label=K.round(K.clip(y_pred, 0, 1))

    return K.mean(K.sum(y_true*y_label,axis=1))
def label_sum(y_true,y_pred):
    y_label=K.round(K.clip(y_pred, 0, 1))

    return K.mean(K.sum(y_label,axis=1))
def true_sum(y_true,y_pred):
    return K.mean(K.sum(y_true,axis=1))
def f1(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    smooth=1.0
    y_label = K.round(K.clip(y_pred, 0, 1))
    y_tp = K.round(K.clip(y_true * y_label, 0, 1))
    y_tp_sum=K.sum(y_tp,axis=1)
    y_true_sum=K.sum(y_true,axis=1)
    y_label_sum=K.sum(y_label,axis=1)
    print(y_true_sum.shape)
    print(y_tp_sum.shape)
    print(y_label_sum.shape)
    dice_array=(y_tp_sum*2+smooth)/(y_label_sum+y_true_sum+smooth)
    dice = K.mean(dice_array)
    return dice

def dice_coef(y_true, y_pred):
    smooth=1.0
    y_tp = y_true*y_pred
    y_tp_sum = K.sum(y_tp, axis=1)
    y_true_sum = K.sum(y_true, axis=1)
    y_label_sum = K.sum(y_pred, axis=1)
    dice_array = (y_tp_sum * 2 + smooth) / (y_label_sum + y_true_sum + smooth)
    dice = K.mean(dice_array)
    return dice
def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true,y_pred)


