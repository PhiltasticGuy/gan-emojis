import os, logging

from PIL import Image

from matplotlib import pyplot

from numpy import asarray
from numpy import savez_compressed
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout

# Display number of available GPUs on the local machine
import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

# List locals devices available for TensorFlow
from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())

# Enable debug logs for TensorFlow
# tf.debugging.set_log_device_placement(True)