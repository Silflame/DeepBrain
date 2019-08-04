import cv2
import numpy as np
import utils
import numpy as np
from DataSet import DataSet
from V3_plus import Deeplabv3
import config
def get_heat_map(gt,output,lamda=0.1,mode=0):
    '''

    :param gt: RGB image [512,512,3]
    :param output: sigmoid output [512,512]
    :param lamda: weights of heatmap
    :param mode: mode=0 heatmap+RGB image
                mode =1 heatmap
    :return:
    '''
    heatmap = np.uint8(255 * output)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * lamda + gt
    superimposed_img = superimposed_img / (255 * (1 + lamda)) * 255
    superimposed_img = np.uint8(superimposed_img)
    if (mode==0):
        cv2.imshow('heatmap',superimposed_img)
    if (mode==1):
        cv2.imshow('heatmap',heatmap)
    cv2.waitKey(0)
    return output
if __name__ == '__main__':
    dataSet = DataSet()
    v3 = Deeplabv3(input_shape=(512, 512, 1), classes=1, activation='sigmoid')
    v3.load_weights('v3_std20.h5')
    dice_list = []
    name_list = []
    RGB_image=dataSet.get_test_RGB_image()
    now_image, now_gt, now_name = dataSet.get_test_data_obo(True)
    now_gt=now_gt.reshape([512,512])
    now_output=v3.predict(now_image)
    now_output=now_output.reshape([512,512])
    get_heat_map(RGB_image,now_output,lamda=0.4,mode=0)