import numpy as np 
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 
from PIL import Image 
from tensorflow.keras import layers 
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.optimizers import Adam 
import tensorflow as tf 
import os

base_dir = 'Planes'

img_size = 224
batch = 64

# Create a data augmentor 
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2, 
								zoom_range=0.2, horizontal_flip=True, 
								validation_split=0.2) 

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, 
								validation_split=0.2) 

# Create datasets 
train_datagen = train_datagen.flow_from_directory(base_dir, 
												target_size=( 
													img_size, img_size), 
												subset='training', 
												batch_size=batch) 
test_datagen = test_datagen.flow_from_directory(base_dir, 
												target_size=( 
													img_size, img_size), 
												subset='validation', 
												batch_size=batch) 




model = Sequential() 
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', 
				activation='relu', input_shape=(224, 224, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2))) 


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
				padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
				padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 

model.add(Conv2D(filters=64, kernel_size=(3, 3), 
				padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 

model.add(Flatten()) 
model.add(Dense(512)) 
model.add(Activation('relu')) 
model.add(Dense(9, activation="softmax")) 


model.summary()



model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy']
              )


epochs=9
model.fit(train_datagen,epochs=epochs,validation_data=test_datagen)

model.save('recognition.model')