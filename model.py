import csv
import cv2
import fnmatch
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
import math
from matplotlib import pyplot
import numpy as np
import os
import scipy
from sklearn.utils import shuffle

DRIVING_LOG_PATH = './data'
DRIVING_LOG_FILE = 'driving_log.csv'

IMAGE_PATH = './data/IMG'

WIDTH = 66
LENGTH = 200
DEPTH = 1

SAMPLES_PER_EPOCH = 512
EPOCH = 4
VALIDATION_SET_SIZE = 102

def generate_training_sample():
    file = open(os.path.join(DRIVING_LOG_PATH, DRIVING_LOG_FILE), 'r')
    reader = csv.reader(file)
    
    reader.__next__()
    
    yield from generate_sample(reader)
    
    file.close()

def generate_validation_sample():
    file = open(os.path.join(DRIVING_LOG_PATH, DRIVING_LOG_FILE), 'r')
    reader = csv.reader(file)
    reader = reversed(list(reader))
    
    yield from generate_sample(reader)
    
    file.close()

def generate_sample(reader):
    while True:
        line = reader.__next__()
        
        path = os.path.join(
            IMAGE_PATH,
            line[0].strip('IMG/')
        )
        image = cv2.imread(path)
        y_start = 64
        y_end = image.shape[0] - 30
        x_start = 60
        x_end = image.shape[1] - 60
        image = image[y_start:y_end, x_start:x_end]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(
            image,
            dtype = 'float32'
        )
        image = image / 255
        image = image.reshape(
            1,
            image.shape[0],
            image.shape[1],
            DEPTH
        )
        steering_angle = np.array(
            line[3], 
            dtype = 'float32')
        steering_angle = steering_angle.reshape(1, 1)
        
        yield (image, steering_angle)

convolution_filter = 24
kernel_size = 5
stride_size = 2

model = Sequential()
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid',
    subsample = (stride_size, stride_size),
    input_shape = (WIDTH, LENGTH, DEPTH)
))

convolution_filter = 36

model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid',
    subsample = (stride_size, stride_size)
))

convolution_filter = 48

model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid',
    subsample = (stride_size, stride_size)
))

convolution_filter = 64
kernel_size = 3

model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid'
))
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid'
))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

adam = Adam(lr = 0.000001)

model.compile(optimizer = adam, loss = 'mse')
history = model.fit_generator(
    generate_training_sample(),
    samples_per_epoch = SAMPLES_PER_EPOCH,
    nb_epoch = EPOCH,
    verbose = 1,
    validation_data = generate_validation_sample(),
    nb_val_samples = VALIDATION_SET_SIZE
)

model_json = model.to_json()

json_file = open('model.json', 'w')
json_file.write(model_json)

model.save_weights('model.h5')
