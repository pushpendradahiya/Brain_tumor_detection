#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import os
import numpy as np
# import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
from skimage import io
from sklearn.utils import shuffle


# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback

# In[4]:


def find_images(path_dir, path_nor, path_t):
    num=50
    X_train=np.zeros(((274+109)*num+842,240,240),dtype=np.uint16)
    Y_train=np.zeros(((274+109)*num+842,2),dtype=np.uint16)
    j=0
    for item in os.listdir(path_dir):
        item=os.path.join(path_dir,item)
        for item2 in os.listdir(item):
            im ={'T1':None,'gt':None}
            item2=os.path.join(item,item2)
            for item3 in os.listdir(item2):
                item3=os.path.join(item2,item3)
                for item4 in os.listdir(item3):
                    item5=os.path.join(item3,item4)
                    if os.path.isfile(item5) and item5.endswith('.mha'):
                        itk_image = sitk.ReadImage(item5)
                        nd_image = sitk.GetArrayFromImage(itk_image)
                        if 'more' in item5 or 'OT' in item5:
                            im['gt']=nd_image
                        elif 'T1' in item5 and 'T1c' not in item5:
                            im['T1']=nd_image
            for i in range(70,70+num):
                if not sum(sum(im['gt'][i,:,:])): #sum=0 normal brains
                    Y_train[j][0]=1
                else:
                    Y_train[j][1]=1
                X_train[j]=im['T1'][i,:,:]
                j+=1
    for item in os.listdir(path_nor):
        item2 = os.path.join(path_nor,item)
        img = nib.load(item2)
        data=img.get_fdata()
        for i in range(90,90+num):
            Y_train[j][0]=1
            X_train[j]=resize(data[50:210,i,:],(240,240))
            j+=1
    for item in os.listdir(path_t):
        item2=os.path.join(path_t,item)
        for item3 in os.listdir(item2):
            item5=os.path.join(item2,item3)
            if os.path.isfile(item5) and item5.endswith('.jpg'):
                image = io.imread(item5,as_gray=True)
                image = resize(image,(240,240))
                X_train[j]=image
                if item == 'abnormalsJPG':
                    Y_train[j][1]=1
                elif item == 'normalsJPG':
                    Y_train[j][0]=1
                j+=1
    return X_train,Y_train

# In[5]:


path_dir="../BRATS2015_Training"
path_nor="../nii"
path_t="../normalsVsAbnormalsV1"
X_train,Y_train=find_images(path_dir, path_nor, path_t)


# In[15]:

X_train=X_train-np.mean(X_train,axis=0)
np.save("mean",np.mean(X_train,axis=0))
shape=X_train.shape
X_train=X_train.reshape(shape[0],shape[1],shape[2],1)


# In[16]:


np.random.seed(1000)
model=Sequential()

model.add(Conv2D(filters=96, input_shape=(240,240,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[18]:

X_train, Y_train = shuffle(X_train, Y_train)
model_checkpoint = ModelCheckpoint('./Alexnet_brat.hdf5', monitor='loss',verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)
callbacks = [reduce_lr, model_checkpoint]
#model.load_weights("./Alexnet_brat.hdf5")
model.fit(X_train,Y_train,batch_size=32,epochs=200,verbose=1,validation_split=0.2,shuffle=True, callbacks=callbacks)


# In[12]:


#plt.imshow(X_train[13699])


# In[14]:


# Y_train[13699]

