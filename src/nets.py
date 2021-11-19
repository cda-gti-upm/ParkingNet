# Neural network architecture

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import NASNetLarge
from common_flags import FLAGS
from data_generator import get_no_outputs


def get_ramification(no_outputs, ramification_layer):  
    
    # Output dimension (empty place probability)
    output_dim = 1
    
    outputs_list = []    
    for i in range(no_outputs):
        output_n = Dense(output_dim)(ramification_layer)
        output_n = Activation('sigmoid', name='a' + str(i) )(output_n)
        outputs_list.append(output_n)
    
    return outputs_list    

def get_model(img_height, img_width, img_channels): 

    img_input = Input(shape=(img_height, img_width, img_channels)) 
    
    #CONVOLUTIONAL BASE
    if (FLAGS.conv_base == "Resnet50"):
        model = ResNet50( include_top=False, weights='imagenet',  input_tensor = img_input)         
    if (FLAGS.conv_base == "Xception"):
        model = Xception( include_top=False, weights='imagenet', input_tensor = img_input)
    if (FLAGS.conv_base == "NASNetLarge"):
        model = NASNetLarge( include_top=False, weights='imagenet', input_tensor=img_input)
    
    x = model.output
    
    # FC layers
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.5)(x)    
   
    # Get number of outputs and ramification of the network
    no_outputs = get_no_outputs()    
    outputs_list = get_ramification(no_outputs, x)
       
    # Model
    model = Model(inputs=[img_input], outputs=outputs_list)
    #print(model.summary())
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, decay=1e-6)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, 
                  metrics=['binary_accuracy'], 
                  loss_weights=np.ones((no_outputs,)).tolist() )      
    
    return model









    
    