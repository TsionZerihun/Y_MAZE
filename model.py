#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.losses import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import *
from tensorflow.keras.preprocessing.image import *

os.environ["CUDA_VISIBLE_DEVICES"]="5"

BATCH_SIZE=16

color=np.array([(0,255,255),(0,255,255),(0,255,255),(0,255,255),(255,0,0),(0,255,0),(0,0,255)], dtype=np.float32)

imgsize=(512,512)

testimg=tf.image.resize_with_pad(cv2.imread("YMTest.jpg"),imgsize[1],imgsize[0])


# In[4]:


def visualize(x,y,Y=None):
    x=np.array(x).copy()
    y=np.array(y).copy()
    H,W=x.shape[:2]
    y=y*(W,H)
    
    for i,ty in enumerate(y):
        cv2.circle(x, tuple(map(int,ty)), 5, tuple(map(int,color[i])), -1)
    
    return x


# In[5]:


class LayerWise_CartPool2D(tf.keras.layers.Layer):
    def __init__(self,kernel_size=3, strides=1, padding='VALID', dilation_rate=1,*args,**kwargs):
        super(LayerWise_CartPool2D, self).__init__(*args,**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        
    def build(self, input_shape):
        self.batch, self.height, self.width, self.channel=input_shape

    def call(self, inputs):
        if self.batch is not None:
            batch=tf.shape(inputs)[0]
            matrice_kernel=tf.image.extract_patches(inputs,[1,self.kernel_size,self.kernel_size,1],[1,self.strides,self.strides,1],[1,self.dilation_rate,self.dilation_rate,1],padding=self.padding)
            matrice_kernel=tf.reshape(matrice_kernel,tf.concat([[batch],tf.shape(matrice_kernel)[1:3],[1],[self.kernel_size,self.kernel_size,-1]],axis=-1))

            W,H=tf.meshgrid(tf.range(self.kernel_size,dtype=tf.float32),tf.range(self.kernel_size,dtype=tf.float32))
            IW,IH=tf.meshgrid(tf.range(self.width,dtype=tf.float64),tf.range(self.height,dtype=tf.float64))
            e=tf.exp(matrice_kernel)
            s=tf.reduce_sum(e,axis=(-3,-2),keepdims=True)
            g=e/s

            axis_X=(tf.reduce_sum(g*tf.reshape(W,(1,1,1,1,self.kernel_size,self.kernel_size,1)),axis=(-4,-3,-2))-self.kernel_size//2+tf.reshape(IW,(1,self.height,self.width,1)))/self.width
            axis_Y=(tf.reduce_sum(g*tf.reshape(H,(1,1,1,1,self.kernel_size,self.kernel_size,1)),axis=(-4,-3,-2))-self.kernel_size//2+tf.reshape(IH,(1,self.height,self.width,1)))/self.height

            return tf.concat([axis_X,axis_Y], axis=-1)
        else:
            return tf.concat([inputs,inputs], axis=-1)
    
    def get_config(self):
        return super().get_config().copy()


# In[6]:


model=tf.keras.models.load_model("ymaze.h5",custom_objects={"LayerWise_CartPool2D":LayerWise_CartPool2D}, compile=False)


# In[7]:


testout=visualize(testimg,np.array(model(np.array([testimg]))[0]))[...,::-1]


# In[14]:


plt.imshow(testout)


# ## Predict from video

# In[9]:


cap=cv2.VideoCapture('ymaze.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:break
    
    img=tf.image.resize_with_pad(frame,512,512)
    model_out=np.array(model(np.array([img]))[0]) #Here's coordinates
    a=(model_out-np.array([[0., (1/1-9/16)/2]]))*np.array([[1., (16/9)]]) #Here's coordinates transformed as video's ratio
    img=visualize(frame,a).astype(np.uint8) #Here's figured-out prediction
    
cap.release()


# ## Make as video

# In[10]:


cap=cv2.VideoCapture('ymaze.avi')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('ymaze.avi', fourcc, 29.7, (1280,720))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:break
    
    img=tf.image.resize_with_pad(frame,512,512)
    model_out=np.array(model(np.array([img]))[0]) #Here's coordinate <- model(~)[0]'s output
    a=(model_out-np.array([[0., (1/1-9/16)/2]]))*np.array([[1., (16/9)]])
    img=visualize(frame,a).astype(np.uint8)
    
    out.write(img)
    
out.release()
cap.release()


# In[16]:


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('1ymaze_r.avi', fourcc, 29.7, (1280,720))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:break
    
    out.write(frame[...,::-1])
    
out.release()
cap.release()

