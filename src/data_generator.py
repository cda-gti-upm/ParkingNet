# Data generator

import pandas as pandas
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from common_flags import FLAGS


# This function returns the batch: (inputs, targets)               
def load_data(directory, idx, batch_size):
    parking_import = pandas.read_csv(directory + "/images/groundtruth.txt",
        skiprows=idx*batch_size, nrows=batch_size, header=None, delim_whitespace=True)
    parking_data = np.array(parking_import.values)

    batch_images = []
    batch_labels = []  
    for i in range(len(parking_data)):
        image_name = parking_data[i][0]            
        image_name = directory + "/images/" + image_name
        
        img = load_img(image_name, color_mode="rgb", target_size=(FLAGS.img_height, FLAGS.img_width), interpolation="nearest") 
        img = img_to_array(img)
        # Normalization
        img = img / 255.0
        img = img.astype("float32")       

        batch_images.append(img)         
        batch_labels.append(np.array(parking_data[i][1:]))
        
    batch_images = np.array(batch_images).astype('float32')
    batch_labels = np.array(batch_labels).T    
    batchlist = []
    for element in batch_labels:
        batchlist.append(np.array(element).astype('float32') )        

    return batch_images, batchlist

# This function returns a generator yielding batches of (batch_images, batch_labels).
def batch_generator(directory, batch_size, steps):    
    idx = 1
    while True:
        # Yields batches
        yield load_data(directory, idx-1, batch_size)

        if idx < steps:
            idx += 1
        else:
            idx = 1

# Loads an image to make predictions 
def load_prediction(image):  
          
    img = load_img(image, color_mode="rgb", target_size=(FLAGS.img_height, FLAGS.img_width), interpolation="nearest") 
    img = img_to_array(img)                 
    
    img = np.array(img)
    img =  img[None,:,:,:]
    img = img / 255.0    
    
    return img 

# Read GT file and return the number of images
def get_no_images(dir):
    full_path = FLAGS.data_route + "/" + dir +"/images/groundtruth.txt"
    parking_import = pandas.read_csv(full_path, delim_whitespace=True)
    no_images = parking_import.shape[0]
    print("Found " + str(no_images) + " images in " + full_path)
    return no_images

# Read GT file and return the number of outputs
def get_no_outputs():
    parking_import = pandas.read_csv( FLAGS.data_route + "/train/images/groundtruth.txt", delim_whitespace=True)
    no_outputs = parking_import.shape[1] - 1
       
    return no_outputs
