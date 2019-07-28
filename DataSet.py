import cv2
import numpy as np
from PIL import Image
import os
import config
class DataSet:
    def __init__(self,image_path='./images',label_path='./label'):
        print('load dataset')
        self.image_path=image_path
        self.label_path=label_path
        self.image_list=self.get_file_list(self.image_path)
        self.label_list=self.get_file_list(self.label_path)
        self.train_list=self.get_list_fromfile('train.txt')
        print(self.train_list.shape)
        self.val_list=self.get_list_fromfile('val.txt')
        self.test_list=self.get_list_fromfile('test.txt')
        self.train_loc=0
        self.val_loc=0
        self.test_loc=0
        print('dataset load complete')
    def get_train_dataBatch(self,batchsize=8):
        '''
            get a batch data
            train_loc is set to go over all dataset (don't worry)
            output image,label
            image dim : [batch,512,512,3]
            label dim: [batch,512,512] all value is int belongs to {0,1}
        '''
        n=self.train_list.shape[0]
        end = min(self.train_loc+batchsize,n)
        start = self.train_loc
        batch_list = self.train_list[start:end]
        if (end-start!=batchsize):
            need=batchsize-end+start
            batch_list=np.concatenate([batch_list,self.train_list[0:need]])
        self.train_loc = (self.train_loc + batchsize) % n
        train_image=[]
        train_label=[]
        for x in range(0,batchsize):
            name=str(batch_list[x])
            now_image=self.get_image(self.image_path+'/'+name+'.jpg')
            now_label=self.get_label(self.label_path+'/'+name+'.png')
            train_image.append(now_image)
            train_label.append(now_label)
        train_label=np.array(train_label).reshape([batchsize,-1])
        return np.array(train_image),train_label
    def get_val_dataBatch(self,batchsize=8):
        '''
            get a batch data
            output image,label
            image dim : [batch,512,512,3]
            label dim: [batch,512,512] all value is int belongs to {0,1}
        '''
        n=self.val_list.shape[0]
        end = min(self.val_loc+batchsize,n)
        start = self.val_loc
        batch_list = self.val_list[start:end]
        if (end-start!=batchsize):
            need=batchsize-end+start
            batch_list=np.concatenate([batch_list,self.val_list[0:need]])
        self.val_loc = (self.train_loc + batchsize) % n
        val_image=[]
        val_label=[]
        for x in range(0,batchsize):
            name=str(batch_list[x])
            now_image=self.get_image(self.image_path+'/'+name+'.jpg')
            now_label=self.get_label(self.label_path+'/'+name+'.png')
            val_image.append(now_image)
            val_label.append(now_label)
        val_label=np.array(val_label).reshape([batchsize,-1])
        return np.array(val_image),val_label
    def get_test_data_obo(self):
        '''
            get test_data one by one
            use in evluation
            output image,label
            image dim : [1,512,512,3] (to predict,so add a axis in axis=0)
            label dim: [512*512] all value is int belongs to {0,1}
        '''
        n = self.test_list.shape[0]
        now_name = self.test_list[self.test_loc]
        self.test_loc+=1
        assert self.test_loc <= n
        now_image=self.get_image(self.image_path+'/'+now_name+'.jpg')
        now_label=self.get_label(self.label_path+'/'+now_name+'.png')
        now_label=now_label.reshape([-1])
        now_image=now_image[np.newaxis,:,:,:]
        return now_image,now_label
    def visualize(self,image,label=False):
        '''
            image a numpy array can be get from train/val list
            input dim should be [512,512,3] for image or [512,512] for label
            because label value is set to be 1
            to visualize it I set it to be 128
            output a picture
            if you want to visualize a label image set parameter label=True
        '''
        if (label==True):
            image[image==1]=128
            image=image.reshape([config.img_width,config.img_height])
        cv2.imshow('image',image)
        cv2.waitKey(0)
    def get_list_fromfile(self,filename):
        f=open(filename,'r')
        returnlist=[]
        line = None
        while True:
            line=f.readline().replace('\n', '')
            if (line is None) or len(line)==0 : break
            returnlist.append(int(line))
        f.close()
        return np.array(returnlist)

    def get_file_list(self,path):
        path_list=os.listdir(path)
        return np.array(path_list)
    def get_image(self,img_path):
        image=cv2.imread(img_path)
        return image
    def get_label(self,label_path):
        label=cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        label[label!=0]=1
        return label

if  __name__ == '__main__':
    dataSet=DataSet()
    '''
    image_path=dataSet.get_file_list(dataSet.image_path)
    label_path=dataSet.get_file_list(dataSet.label_path)
    print(image_path[1])
    print(len(image_path))
    image=dataSet.get_image(dataSet.image_path+'/'+image_path[2])
    label=dataSet.get_label(dataSet.label_path+'/'+label_path[2])
    print(image)
    print(image.shape)
    print(label.shape)
    print(label_path[2])
    print(label.tolist())

    dataSet.visualize(image)
    dataSet.visualize(label,True)

    val_array=dataSet.get_list_fromfile('val.txt')
    print(val_array.shape)
    print(val_array)
    '''
    img_list,label_list=dataSet.get_train_dataBatch()
    print(img_list.shape)
    print(label_list.shape)
    dataSet.visualize(label_list[2],True)
    dataSet.visualize(img_list[2])
    val_list,val_label=dataSet.get_val_dataBatch()
    dataSet.visualize(val_list[0])
    dataSet.visualize(val_label[0],True)


