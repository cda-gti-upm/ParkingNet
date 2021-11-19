# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:37:08 2021

@author: jorge
"""

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.applications.resnet50 import ResNet50
import pandas as pandas
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt



img_height = 200
img_width = 267
img_channels = 3

learning_rate = 1e-7
batch_size = 5
database_len_train = 1700
database_len_val = 100

nb_epoch = 20
# Number of batches
steps_per_epoch = np.ceil( database_len_train / batch_size )
validation_steps = np.ceil( database_len_val / batch_size )



#route = "/content/drive/MyDrive/TFM/data" 
#route = "/content/drive/MyDrive/TFM/data_reduced" 
route = "data_reduced" 

# This function must return the generator ->
# generator : a generator whose output must be a list of the form: (inputs, targets)               
def load_data(Train_df, idx, batch_size):
        
    parking_import = pandas.read_csv( Train_df + "/groundtruth.txt", skiprows=idx*batch_size, nrows=batch_size, delim_whitespace=True)
    parking_data = np.array(parking_import.values)

    batch_images = []
    batch_labels = []
    
    #route = "/content/drive/MyDrive/ML_PROJECT/images_equal/" 
    #route = "/content/drive/MyDrive/TFM/data_reduced"  
    

    for i in range (len(parking_data)):    
        image_name = parking_data[i][0]            
        image_name =  Train_df + "/images/" + image_name        
        img = load_img(image_name, color_mode="rgb", target_size=(img_height, img_width), interpolation="nearest")     
        img = img_to_array(img)        
        batch_images.append(img)    
        #batch_labels.append(parking_data[i][1])
        batch_labels.append(parking_data[i][1:])
    
    batch_images = np.array(batch_images)
    batch_images = batch_images / 255.0
    batch_labels = np.array(batch_labels).astype('float32')
    
    return (batch_images, batch_labels)           


def batch_generator(Train_df, batch_size, steps):    
    idx = 1
    while True:
        
        yield load_data(Train_df, idx-1, batch_size)## Yields data
        
        if idx < steps:
            idx+=1
        else:
            idx=1
            
### Generator objects for train and validation
my_training_batch_generator = batch_generator( route + "/train", batch_size, steps_per_epoch)
my_validation_batch_generator = batch_generator(route + "/val", batch_size, validation_steps)



##################### RESNET 50 MODEL #########################################
img_input = Input(shape=(img_height, img_width, img_channels)) 


model = ResNet50( include_top=False, weights='imagenet',  input_tensor = img_input) 
#model = keras.applications.Xception( include_top=False, weights='imagenet', input_tensor=img_input)
#model = keras.applications.NASNetLarge( include_top=False, weights='imagenet', input_tensor=img_input)

x = model.output

# FC layers
x = Flatten()(x)
x = Dense(1024)(x)
x = Activation('sigmoid')(x)
x = Dropout(0.5)(x)

# Output dimension (empty place probability)
output_dim = 1

x1 = Dense(output_dim)(x)
x1 = Activation('sigmoid', name='a1')(x1)

x2 = Dense(output_dim)(x)
x2 = Activation('sigmoid', name='a2')(x2)
  
x3 = Dense(output_dim)(x)
x3 = Activation('sigmoid', name='a3')(x3)
  
x4 = Dense(output_dim)(x)
x4 = Activation('sigmoid', name='a4')(x4)
  
x5 = Dense(output_dim)(x)
x5 = Activation('sigmoid', name='a5')(x5)
  
x6 = Dense(output_dim)(x)
x6= Activation('sigmoid', name='a6')(x6)
  
x7 = Dense(output_dim)(x)
x7 = Activation('sigmoid', name='a7')(x7)
  
x8 = Dense(output_dim)(x)
x8 = Activation('sigmoid', name='a8')(x8)
  
x9 = Dense(output_dim)(x)
x9 = Activation('sigmoid', name='a9')(x9)
  
x10 = Dense(output_dim)(x)
x10 = Activation('sigmoid', name='a10')(x10)
  
x11 = Dense(output_dim)(x)
x11 = Activation('sigmoid', name='a11')(x11)
  
x12 = Dense(output_dim)(x)
x12 = Activation('sigmoid', name='a12')(x12)
  
x13= Dense(output_dim)(x)
x13 = Activation('sigmoid', name='a13')(x13)
  
x14 = Dense(output_dim)(x)
x14 = Activation('sigmoid', name='a14')(x14)
  
x15 = Dense(output_dim)(x)
x15 = Activation('sigmoid', name='a15')(x15)
  
x16 = Dense(output_dim)(x)
x16 = Activation('sigmoid', name='a16')(x16)
  
x17 = Dense(output_dim)(x)
x17 = Activation('sigmoid', name='a17')(x17)
  
x18 = Dense(output_dim)(x)
x18 = Activation('sigmoid', name='a18')(x18)
  
x19 = Dense(output_dim)(x)
x19 = Activation('sigmoid', name='a19')(x19)
  
x20 = Dense(output_dim)(x)
x20 = Activation('sigmoid', name='a20')(x20)
  
x21 = Dense(output_dim)(x)
x21 = Activation('sigmoid', name='a21')(x21)

model = Model(inputs=[img_input], outputs=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21])
print(model.summary())





optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
                          
history = model.fit( my_training_batch_generator,
                    epochs = nb_epoch,
                    steps_per_epoch = steps_per_epoch,
                    verbose = 1, 
                    validation_data = my_validation_batch_generator,
                    validation_steps = validation_steps)

