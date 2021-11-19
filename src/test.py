# -*- coding: utf-8 -*-

import numpy as np
import gflags
import nets
import sys
from common_flags import FLAGS
from data_generator import batch_generator, get_no_images
import os



def test():    
    
    # Get a new model
    model = nets.get_model(FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels)

    # Load saved weights
    checkpoint_path = os.path.join('experiments/'+ FLAGS.checkpoint_name + '/checkpoints/')
    model.load_weights(checkpoint_path)
    

    # Test generator
    test_steps = np.ceil( get_no_images('test') / FLAGS.batch_size ) 
    test_batch_generator = batch_generator( FLAGS.data_route + "/test", FLAGS.batch_size, test_steps)    
    
    # EVALUATION
    results = model.evaluate(test_batch_generator,
                              verbose=1,
                              sample_weight=None,
                              steps=test_steps,
                              # callbacks=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False,
                              return_dict=False )
    
    print (" ########################### EVALUATION DONE ################")

    
    


    
def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    test()

if __name__ == "__main__":
   main(sys.argv)
    
    
