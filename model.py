# Import libraries
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import cv2
import fnmatch
import math
import numpy as np
import os
import scipy

# Set parameters
DEBUG = True

PATH = './data'
DRIVING_LOG_FILE = 'driving_log.csv'
BATCH_SIZE = 32

IMAGE_WIDTH = 160
IMAGE_LENGTH = 320
IMAGE_DEPTH = 3

ANGLE_MODIFIER = 0.2

CROP_TOP = 64
CROP_BOTTOM = 30

LEARNING_RATE = 0.000001

EPOCH = 2
VERBOSITY = 2

# Get data
samples = []
with open(os.path.join(PATH, DRIVING_LOG_FILE), 'r') as file:
    reader = csv.reader(file)
    reader.__next__()
    for line in reader:
        samples.append(line)

if DEBUG:
    shuffle(samples)
    samples = samples[:320]

train_set, validation_set = train_test_split(samples, test_size = 0.2)
print('Train set size:', len(train_set))
print('Validation set size:', len(validation_set))

def generate_sample(samples, batch_size = BATCH_SIZE):
    sample_count = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, sample_count, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                path = os.path.join(PATH, batch_sample[0].strip())
                center_image = cv2.imread(path)
                # flip_center_image = cv2.flip(center_image, 1)
                center_image = transform_image(center_image)
                # flip_center_image = transform_image(flip_center_image)
                path = os.path.join(PATH, line[1].strip())
                left_image = cv2.imread(path)
                # flip_left_image = cv2.flip(left_image, 1)
                left_image = transform_image(left_image)
                # flip_left_image = transform_image(flip_left_image)
                path = os.path.join(PATH, line[2].strip())
                right_image = cv2.imread(path)
                # flip_right_image = cv2.flip(right_image, 1)
                right_image = transform_image(right_image)
                # flip_right_image = transform_image(flip_right_image)
                images.extend([
                    center_image,
                    # flip_center_image,
                    left_image,
                    # flip_left_image,
                    right_image
                    # flip_right_image
                ])

                center_angle = np.array(line[3], dtype = 'float32')
                center_angle = transform_angle(center_angle)
                # flip_center_angle = transform_angle(center_angle * -1.0)
                left_angle = transform_angle(
                    center_angle,
                    ANGLE_MODIFIER
                )
                # flip_left_angle = transform_angle(left_angle * -1.0)
                right_angle = transform_angle(
                    center_angle,
                    -ANGLE_MODIFIER
                )
                # flip_right_angle = transform_angle(right_angle * -1.0)
                angles.extend([
                    center_angle,
                    # flip_center_angle,
                    left_angle,
                    # flip_left_angle,
                    right_angle
                    # flip_right_angle
                ])

            images = np.array(images, dtype = 'float32')
            angles = np.array(angles, dtype = 'float32')

            yield shuffle(images, angles)

train_generator = generate_sample(train_set)
validation_generator = generate_sample(validation_set)

# Transform data
def transform_image(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image, dtype = 'float32')

    return image.reshape(
        IMAGE_WIDTH,
        IMAGE_LENGTH,
        IMAGE_DEPTH
    )

def transform_angle(steering_angle, modifier = 0.0):
    steering_angle = steering_angle + modifier
    steering_angle = np.array(steering_angle, dtype = 'float32')

    return steering_angle.reshape(1)

# Design model
convolution_filter = 24
kernel_size = 5
stride_size = 2
model = Sequential()
model.add(
    Cropping2D(
        cropping = ((CROP_TOP, CROP_BOTTOM), (0, 0)),
        input_shape = (IMAGE_WIDTH, IMAGE_LENGTH, IMAGE_DEPTH)
    )
)
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid',
    subsample = (stride_size, stride_size)
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
    train_generator,
    samples_per_epoch = len(train_set),
    nb_epoch = EPOCH,
    verbose = VERBOSITY,
    validation_data = validation_generator,
    nb_val_samples = len(validation_set)
)

# Save model
model.save('model.h5')

# Save chart
chart = pyplot.gcf()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.legend(['Training', 'Validation'])
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
chart.savefig('loss.png')