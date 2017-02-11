# Import libraries
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

# Set parameters
PATH = './data'
DRIVING_LOG_FILE = 'driving_log.csv'

WIDTH = 66
LENGTH = 320
DEPTH = 1

STEERING_ANGLE_MODIFIER = 0.2

LEARNING_RATE = 0.000001

SAMPLES_PER_EPOCH = 9642
EPOCH = 2
VERBOSITY = 2
VALIDATION_SET_SIZE = 2409

# Get data
def generate_sample(reader):
    while True:
        line = reader.__next__()
        
        path = os.path.join(
            PATH,
            line[0].strip()
        )
        center_image = cv2.imread(path)
        center_image = transform_image(center_image)

        path = os.path.join(
            PATH,
            line[1].strip()
        )
        left_image = cv2.imread(path)
        left_image = transform_image(left_image)

        path = os.path.join(
            PATH,
            line[2].strip()
        )
        right_image = cv2.imread(path)
        right_image = transform_image(right_image)

        image = np.concatenate((center_image, left_image, right_image))

        center_steering_angle = np.array(
            line[3], 
            dtype = 'float32'
        )
        center_steering_angle = transform_steering_angle(
            center_steering_angle
        )
        left_steering_angle = transform_steering_angle(
            center_steering_angle,
            STEERING_ANGLE_MODIFIER
        )
        right_steering_angle = transform_steering_angle(
            center_steering_angle,
            -STEERING_ANGLE_MODIFIER
        )
        steering_angle = np.concatenate((
            center_steering_angle,
            left_steering_angle,
            right_steering_angle
        ))
        
        yield (image, steering_angle)

def generate_training_sample():
    file = open(os.path.join(PATH, DRIVING_LOG_FILE), 'r')
    reader = csv.reader(file)
    
    reader.__next__()
    
    yield from generate_sample(reader)
    
    file.close()

def generate_validation_sample():
    file = open(os.path.join(PATH, DRIVING_LOG_FILE), 'r')
    reader = csv.reader(file)
    reader = reversed(list(reader))
    
    yield from generate_sample(reader)
    
    file.close()

# Transform data
def transform_image(image):
    y_start = 64
    y_end = image.shape[0] - 30
    x_start = 0
    x_end = image.shape[1]
    image = image[y_start:y_end, x_start:x_end]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(
        image,
        dtype = 'float32'
    )
    image = image / 255
    
    return image.reshape(
        1,
        WIDTH,
        LENGTH,
        DEPTH
    )

def transform_steering_angle(steering_angle, modifier = 0.0):
    steering_angle = steering_angle + modifier
    return steering_angle.reshape(1, 1)

# Design model
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

# Train model
adam = Adam(lr = LEARNING_RATE)
model.compile(optimizer = adam, loss = 'mse')
history = model.fit_generator(
    generate_training_sample(),
    samples_per_epoch = SAMPLES_PER_EPOCH,
    nb_epoch = EPOCH,
    verbose = VERBOSITY,
    validation_data = generate_validation_sample(),
    nb_val_samples = VALIDATION_SET_SIZE
)

# Save model
model_json = model.to_json()
json_file = open('model.json', 'w')
json_file.write(model_json)
model.save_weights('model.h5')

# Save chart
chart = pyplot.gcf()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.legend(['Training', 'Validation'])
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
chart.savefig('loss.png')