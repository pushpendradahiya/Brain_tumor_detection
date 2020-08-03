
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, Callback,ReduceLROnPlateau
from keras import backend as keras
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import h5py


# In[2]:


def unet(pretrained_weights = None,input_size = (240,240,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

#     drop5 = (UpSampling2D(size = (2,2))(drop5))
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


# In[3]:


def convert_dims(x,y):
    x = x[:,:,1,:,:]
    y = 1 - y[:,:,0]
    xshape = x.shape
    x = np.reshape(x,(xshape[0]*xshape[1],xshape[2],xshape[3]))
    y = np.reshape(y,(len(y),80,72,64))
    y = np.transpose(y,(0,3,1,2))
    y = np.reshape(y,(xshape[0]*xshape[1],xshape[2],xshape[3]))
    x = np.expand_dims(x,axis = 3)
    y = np.expand_dims(y,axis = 3)
    return x,y


# In[15]:


epoch_frames = 0
def visualize(im, actual_target, predicted_target):
    im = im[0,:,:,0]
    actual_target = actual_target[0,:,:,0]
    predicted_target = predicted_target[0,:,:,0]
    
    plt.figure(1,(10,4))
    plt.subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(im, cmap = "gray") # orientation='horizontal',fraction=0.046, pad=0.04
#     plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(actual_target, cmap = "magma",vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(predicted_target, cmap = "magma",vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.axis('off')
    t = 'Class: ' + str(int(np.sum(predicted_target.ravel()) > 10))
    plt.text(0.5, 0.5, t, ha = 'center',va='top',wrap=True)
    global epoch_frames
    epoch_frames += 1
    plt.suptitle('epoch: ' + str(epoch_frames))
    plt.savefig("run6/epoch_frames/frame"+str(epoch_frames)+ '.png')
#     plt.show(block=False)
    plt.close()

# Custom callback
class plot_figure(Callback):
    def on_train_begin(self, logs={}):
        self.predicted_label = []
        # d = home +'/output/epoch_frames/'
        # t=[os.remove(f) for f in [os.path.join(d,f) for f in os.listdir(d)]]    
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.predicted_label = self.model.predict(test_data, verbose=0)
        visualize(test_data, test_label, self.predicted_label)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

plot_callback = plot_figure()
model_checkpoint = ModelCheckpoint('./run6/unet_brat.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.8, patience=3, min_lr=0.00001, verbose=1)
callbacks = [reduce_lr, plot_callback,model_checkpoint]


# In[5]:


hf = h5py.File('data.hdf5', 'r')
train_x = np.array(hf.get('train_x'))
train_y = np.array(hf.get('train_y'))
val_x = np.array(hf.get('valid_x'))
val_y = np.array(hf.get('valid_y'))
test_x = np.array(hf.get('test_x'))
test_y = np.array(hf.get('test_y'))


# In[21]:


train_data,train_label = convert_dims(train_x,train_y)
val_data,val_label = convert_dims(val_x,val_y)
test_data1,test_label1 = convert_dims(test_x,test_y)


# In[24]:


idx=20
test_data,test_label = test_data1[idx:idx+1],test_label1[idx:idx+1]
# visualize(test_data, test_label, test_label)


# In[8]:


model = unet(input_size = train_data[0].shape)


# In[25]:


history = model.fit(train_data, train_label, batch_size=32,
          epochs=100, verbose=1,
          shuffle=True, callbacks=callbacks,
         validation_data = (val_data,val_label)
         )

try:
	f = open('./run6/history.txt','w')
	f.write('ACC:\t\t\t\n')
	f.write(str(history.history['acc']))
	f.write('LOSS:\t\t\t')
	f.write('\n')
	f.write(str(history.history['loss']))
	f.write('VAL_LOSS:\t\t\t')
	f.write('\n')
	f.write(str(history.history['val_loss']))
	f.write('VAL_ACC:\t\t\t')
	f.write('\n')
	f.write(str(history.history['val_acc']))
	f.close()
except:
	print('Not writng to history.txt')
	pass


# ### Deprecated Code 

# In[ ]:


# count = 0
# class_1 = []
# for i in range(len(Y_train)):
#     s = np.sum(Y_train[i].ravel())
# #     print(i,s)
#     if s == 0:
#         class_1.append(i)
#         count+=1

# print(count,class_1)
# shape = X_train.shape
# x_train = np.zeros((512,shape[1],shape[2],1))
# y_train = np.zeros((512,shape[1],shape[2],1))
# for i in range(512):
#     if i < 274:
#         x_train[i] = X_train[i]
#         y_train[i] = Y_train[i]
# cnt = 274
# for nidx in class_1:
#     for j in range(13):
#         x_train[cnt] = X_train[nidx]
#         y_train[cnt] = Y_train[nidx]
#         cnt+=1
# for i in range(4):
#     x_train[cnt] = X_train[1]
#     y_train[cnt] = Y_train[1]
#     cnt+=1
# y_train.shape


# In[ ]:


# # import SimpleITK as sitk
# def find_images(path_dir):
#     X_train=np.zeros((274*1,240,240),dtype=np.uint16)
#     Y_train=np.zeros((274*1,240,240),dtype=np.uint16)
#     j=0
#     for item in os.listdir(path_dir):
#         item=os.path.join(path_dir,item)
#         for item2 in os.listdir(item):
#             im ={'T1':None,'gt':None}
#             item2=os.path.join(item,item2)
#             for item3 in os.listdir(item2):
#                 item3=os.path.join(item2,item3)
#                 for item4 in os.listdir(item3):
#                     item5=os.path.join(item3,item4)
#                     if os.path.isfile(item5) and item5.endswith('.mha'):
#                         itk_image = sitk.ReadImage(item5)
#                         nd_image = sitk.GetArrayFromImage(itk_image)
#                         if 'more' in item5 or 'OT' in item5:
#                             im['gt']=nd_image
#                         elif 'T1' in item5:
#                             im['T1']=nd_image
#             for i in range(80,81):
#                 Y_train[j]=np.where(im['gt'][i,:,:] > 0, 1, 0)
#                 X_train[j]=im['T1'][i,:,:]
#                 j+=1
#     return X_train,Y_train

# path_dir="../BRATS2015_Training"
# X_train,Y_train=find_images(path_dir)
# X_train=X_train-np.mean(X_train,axis=0)
# X_train = np.expand_dims(X_train,axis=4)
# Y_train = np.expand_dims(Y_train,axis=4)
# shape=X_train.shape
# X_train=X_train.reshape(shape[0],shape[1],shape[2],1)
# Y_train=Y_train.reshape(shape[0],shape[1],shape[2],1)
# np.save('x.npy',X_train)
# np.save('y.npy',Y_train)


# # In[ ]:


# get_ipython().run_cell_magic('time', '', "X_train = np.load('x.npy')\nY_train = np.load('y.npy')\nX_train.shape,Y_train.shape")


# # In[ ]:


# plt.imshow(X_train[100,:,:,0]);


# # In[ ]:


# img_w = 192
# img_h = 160
# x,y = 30,40
# plt.figure(1,(14,8))
# plt.subplot(1,2,1)
# plt.imshow(X_train[idx,x:x+img_w,y:y+img_h,0],cmap = "seismic");
# plt.subplot(1,2,2)
# plt.imshow(Y_train[idx,x:x+img_w,y:y+img_h,0],cmap = "magma");
# X_train[idx,x:x+img_w,y:y+img_h,0].shape


# # In[ ]:


# train_examples = 20
# input_data_sub_imgs = [X_train[i,x:x+img_h,y:y+img_w] for i in range(0,train_examples)]
# input_label_sub_imgs = [Y_train[i,x:x+img_h,y:y+img_w] for i in range(0,train_examples)]
# train_data = np.array(input_data_sub_imgs)
# train_label = np.array(input_label_sub_imgs)
# train_data.shape


# # In[ ]:


# test_example = 200 # this means 200th examples
# (test_data, test_label) = X_train[test_example:test_example+1,x:x+img_h,y:y+img_w],Y_train[test_example:test_example+1,x:x+img_h,y:y+img_w]
# test_data.shape


# # In[ ]:


# idx = 258
# visualize(X_train[idx:idx+1],Y_train[idx:idx+1],Y_train[idx:idx+1])

