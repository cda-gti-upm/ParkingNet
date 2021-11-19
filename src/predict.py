# -*- coding: utf-8 -*-


import gflags
import nets
import sys
from common_flags import FLAGS
from data_generator import load_prediction
import os


def predict():
    
    # Get a new model
    model = nets.get_model(FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels)

    # Load saved weights
    checkpoint_path = os.path.join('/media/Data/jcp/experiments/'+ FLAGS.checkpoint_name + '/checkpoints/')
    model.load_weights(checkpoint_path)       
    
    print ("Making predictions...") 
    
    predictions = []
    for image in os.listdir(FLAGS.data_route + "/test/images"):
        if (image.endswith(".JPG") or image.endswith(".jpg")):
            name = image
            image = FLAGS.data_route + "/test/images/" + image
            image = load_prediction(image)
            
            prediction = model.predict(image, verbose=0)
            predictions.append( [name, prediction] )
                
    
    print ("Done. Prediction example:") 
    print (predictions[0])  
        
    
    #Write to file 
    with open('predictions/'+ FLAGS.checkpoint_name + "_predictions.txt", "w+") as text_file:        
        
        for i in range (len(predictions)):
            line = []
            im_name = predictions[i][0]
            line.append(im_name)
            
            for j in range(len(predictions[i][1])):
                value = round(predictions[i][1][j][0][0])
                line.append(str(value))               
                
            line = " ".join(line)             
            text_file.write(line + '\n')            

    print ("Predictions written to file.") 




def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    predict()


if __name__ == "__main__":
   main(sys.argv)