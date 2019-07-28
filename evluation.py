from Unet import Unet
import utils
import numpy as np
from DataSet import DataSet
import config
if __name__== '__main__':
    dataSet=DataSet()
    uNet=Unet.build_model()
    uNet.load_weights('unet.h5')
    dice_list=[]
    for x in range(0,config.test_num):
        now_image,now_gt=dataSet.get_test_data_obo()
        now_output=np.round(uNet.predict(now_image).reshape([-1]))
        dice=utils.get_dice(now_gt,now_output)
        dice_list.append(dice)
    dice_list=np.array(dice_list)
    dice=np.mean(dice_list)
    print('dice :%.4f' % dice)

