from tensorflow.python.keras.layers import Conv2D,UpSampling2D,Dense,Flatten,Input,MaxPooling2D,Concatenate,Cropping2D,Add,Reshape,Dropout,ReLU,Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from utils import f1,label_sum,true_sum,tp_sum,dice_loss
import config
class Unet():
    def __init__(self):
        self.img_shape=config.img_shape
    def Con3x3(self,input,filter,name):
        x = Conv2D(filter,3,activation='relu',padding='same',kernel_initializer = 'he_normal',name=name)(input)
        return x
    def Con2x2(self,input,filter,name):
        x = Conv2D(filter, 2, activation='relu', padding='same', kernel_initializer='he_normal', name=name)(input)
        return x
    def Concatenate(self,input,axis=-1):
        x = Concatenate(axis=axis)(input)
        return x
    def build_model(self):
        input=Input(shape=self.img_shape)

        # Block 1
        conv1_1 = self.Con3x3(input,64,'conv1_1')
        conv1_2 = self.Con3x3(conv1_1,64,'conv1_2')
        # output 320x320
        pool1 = MaxPooling2D(name='pool1')(conv1_2)

        # Block 2
        conv2_1 = self.Con3x3(pool1,128,'conv2_1')
        conv2_2 = self.Con3x3(conv2_1,128,'conv2_2')
        #output 160x160
        pool2 = MaxPooling2D(name='pool2')(conv2_2)

        # Block 3
        conv3_1 = self.Con3x3(pool2,256,'conv3_1')
        conv3_2 = self.Con3x3(conv3_1,256,'conv3_2')
        #output 80*80
        pool3 = MaxPooling2D(name='pool3')(conv3_2)

        # Block 4
        conv4_1 = self.Con3x3(pool3,512,'conv4_1')
        conv4_2 = self.Con3x3(conv4_1,512,'conv4_2')
        drop4 =  Dropout(0.5)(conv4_2)
        #output 40*40
        pool4 = MaxPooling2D(name='pool4')(drop4)

        conv5_1 = self.Con3x3(pool4,1024,'conv5_1')
        conv5_2 = self.Con3x3(conv5_1,1024,'conv5_2')
        drop5 = Dropout(0.5)(conv5_2)
        #output 20x20

        up6_1 = UpSampling2D()(drop5)
        up6_2 = self.Con2x2(up6_1,512,'up6')
        #output 40*40
        merge6 = self.Concatenate([up6_2,drop4],axis=-1)

        conv6_1 = self.Con3x3(merge6,512,'conv6_1')
        conv6_2 = self.Con3x3(conv6_1,512,'conv6_2')

        up7_1 = UpSampling2D()(conv6_2)
        up7_2 = self.Con2x2(up7_1,256,'up7')
        #output 80*80
        merge7 = self.Concatenate([conv3_2,up7_2],axis=-1)

        conv7_1 = self.Con3x3(merge7,256,'conv7_1')
        conv7_2 = self.Con3x3(conv7_1,256,'conv7_2')

        up8_1 = UpSampling2D()(conv7_2)
        up8_2 = self.Con2x2(up8_1,128,'up8')
        #output 160*160
        merge8 = self.Concatenate([conv2_2,up8_2],axis=-1)

        conv8_1 = self.Con3x3(merge8,128,'conv8_1')
        conv8_2 = self.Con3x3(conv8_1,128,'conv8_2')

        up9_1 = UpSampling2D()(conv8_2)
        up9_2 = self.Con2x2(up9_1,64,'up9')
        #output 320*320
        merge9 = self.Concatenate([conv1_2,up9_2],axis=-1)

        conv9_1 = self.Con3x3(merge9,64,'conv9_1')
        conv9_2 = self.Con3x3(conv9_1,64,'conv9_2')

        conv_class_1 = self.Con3x3(conv9_2,2,'conv_class_1')
        conv_class_2 = Conv2D(1,1)(conv_class_1)
        conv_class_2 = Activation('sigmoid')(conv_class_2)
        output = Flatten()(conv_class_2)


        model = Model(input,output)

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_loss, metrics=[f1,tp_sum,label_sum,true_sum])
        return model
if __name__ == '__main__':
    model=Unet().build_model()
    model.summary()






