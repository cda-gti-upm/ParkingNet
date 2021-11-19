# Management of input arguments
import gflags


FLAGS = gflags.FLAGS

# Input
gflags.DEFINE_integer('img_height', 200, 'Target Image Heigh')
gflags.DEFINE_integer('img_width', 267, 'Target Image Width')
gflags.DEFINE_integer('img_channels', 3, 'Image Channels')

# Hyperparameters
gflags.DEFINE_float('learning_rate', 1e-4, 'Learning Rate')
gflags.DEFINE_integer('batch_size', 5, 'Batch Size')
gflags.DEFINE_integer('nb_epoch', 1, 'number of epochs')

gflags.DEFINE_string('data_route', "../data_reduced", 'Folder containing database')
gflags.DEFINE_string('conv_base', "Resnet50", 'Convolutional base')

# Load checkpoint
gflags.DEFINE_string('checkpoint_name', "../checkpoint", 'Folder containing the cheickpont')









