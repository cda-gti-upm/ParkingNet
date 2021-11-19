# Training script for ParkingNet
#python train.py --data_route=/media/Data/cda/ETSIT --conv_base=Resnet50 --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python train.py --data_route=/media/Data/cda/ETSIT --conv_base=Xception --nb_epoch=50 --batch_size=16 --learning_rate=1e-4
#python train.py --data_route=/media/Data/cda/ETSIT --conv_base=NASNetLarge --nb_epoch=50 --batch_size=16 --learning_rate=1e-4

# TODO: Update to Tensorflow 2.X
import numpy as np
from time import time, strftime, localtime
import gflags
import os 
import nets
import sys
from common_flags import FLAGS
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from data_generator import batch_generator, get_no_images

# Force CPU computation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train():
    # TODO: definir diferentes data generator por bbdd
    # Number of steps
    train_steps = int(np.floor(get_no_images('train') / FLAGS.batch_size))
    val_steps = int(np.floor(get_no_images('val') / FLAGS.batch_size))

    # Generator objects for train and validation
    train_generator = batch_generator(FLAGS.data_route + "/train", FLAGS.batch_size, train_steps)
    val_generator = batch_generator(FLAGS.data_route + "/val", FLAGS.batch_size, val_steps)
    
    # Get Model
    model = nets.get_model(FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels)
  

    # Write experiment dir where logs and checkpoints will be saved
    strTime = strftime("%Y%b%d_%Hh%Mm%Ss", localtime(time()))
    dataset_name = os.path.basename(os.path.normpath(FLAGS.data_route))
    experiment_dir = '/media/Data/jcp/experiments/'+ FLAGS.conv_base + "_" + dataset_name + "_" + strTime.format(strTime)

    # Tensorboard callback    
    tboard_log_dir = experiment_dir + '/logs'
    tensorboard_callback = TensorBoard(log_dir = tboard_log_dir, histogram_freq=0)  

    #Checkpoint callback
    checkpoint_path = experiment_dir + '/checkpoints/'   
    model_checkpoint_callback = ModelCheckpoint( filepath= checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True)
        
    # TRAIN MODEL
    history = model.fit( train_generator,
                        epochs = FLAGS.nb_epoch,
                        steps_per_epoch = train_steps, 
                        validation_data = val_generator,
                        validation_steps = val_steps,
                        shuffle = True,
                        verbose=1,
                        callbacks=[model_checkpoint_callback, 
                                   tensorboard_callback])  





def main(argv):

    # Utility main to load flags    
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
      
    train()

if __name__ == "__main__":
   main(sys.argv)
   
   