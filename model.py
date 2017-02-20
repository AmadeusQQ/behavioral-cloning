# Import libraries
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import cv2
import numpy as np
import os
import time

# Set parameters
DEBUG = False

DATA_PATH = './data'
DRIVING_LOG_FILE = 'driving_log.csv'
VALIDATION_SET_SIZE = 0.2

IMAGE_WIDTH = 160
IMAGE_LENGTH = 320
IMAGE_DEPTH = 1
# IMAGE_WIDTH = 80
# IMAGE_LENGTH = 160
# IMAGE_DEPTH = 3

ANGLE_MODIFIER = 0.2
CROP_TOP = 64
CROP_BOTTOM = 30
# CROP_TOP = 32
# CROP_BOTTOM = 15

# BATCH_SIZE = 32
BATCH_SIZE = 1
# DROPOUT = 0.0
LEARNING_RATE = 1e-6
EPOCH = 4
VERBOSITY = 2
MODEL_FILE = 'model.h5'

# Get data
samples = []
# print(os.listdir(DATA_PATH))
# exit()
# for path in ['data-1', 'data-2', 'data-3', 'data-4', 'data-5']:
# for path in ['data-1', 'data-2', 'data-3', 'data-5']:
for path in ['data-1', 'data-2', 'data-3']:
# for path in os.listdir(DATA_PATH):
    with open(os.path.join(DATA_PATH, path, DRIVING_LOG_FILE), 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            samples.append(line)

shuffle(samples)
# print(len(samples))
# exit()

if DEBUG:
    # Plot angles
    angles = []
    for path in ['data-1', 'data-2', 'data-3', 'data-5']:
    # for path in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, path, DRIVING_LOG_FILE), 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                angle = float(line[3])
                angles.append(angle)

    angle_chart = pyplot.figure()
    pyplot.hist(angles)
    pyplot.ylabel('Frequency')
    pyplot.xlabel('Angle')
    angle_chart.savefig('angle.png')
    exit()

    samples = samples[:320]

    EPOCH = 1

train_set, validation_set = train_test_split(
    samples,
    test_size = VALIDATION_SET_SIZE
)
samples_per_epoch = len(train_set) / BATCH_SIZE
validation_samples = len(validation_set) / BATCH_SIZE
# samples_per_epoch = len(train_set)
# validation_samples = len(validation_set)

def generate_train_sample(samples, batch_size = BATCH_SIZE):
    sample_count = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, sample_count, batch_size):
            images = []
            angles = []

            batch_samples = samples[offset : offset + batch_size]

            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0].strip())
                flip_center_image = cv2.flip(center_image, 1)
                center_image = transform_image(center_image)
                flip_center_image = transform_image(flip_center_image)
                path = os.path.join(batch_sample[1].strip())
                left_image = cv2.imread(path)
                flip_left_image = cv2.flip(left_image, 1)
                left_image = transform_image(left_image)
                flip_left_image = transform_image(flip_left_image)
                path = os.path.join(batch_sample[2].strip())
                right_image = cv2.imread(path)
                flip_right_image = cv2.flip(right_image, 1)
                right_image = transform_image(right_image)
                flip_right_image = transform_image(flip_right_image)
                images.extend([
                    center_image,
                    flip_center_image,
                    left_image,
                    flip_left_image,
                    right_image,
                    flip_right_image
                ])

                center_angle = np.array(batch_sample[3], dtype = 'float32')
                center_angle = transform_angle(center_angle)
                flip_center_angle = transform_angle(center_angle * -1.0)
                left_angle = transform_angle(
                    center_angle,
                    ANGLE_MODIFIER
                )
                flip_left_angle = transform_angle(left_angle * -1.0)
                right_angle = transform_angle(
                    center_angle,
                    -ANGLE_MODIFIER
                )
                flip_right_angle = transform_angle(right_angle * -1.0)
                angles.extend([
                    center_angle,
                    flip_center_angle,
                    left_angle,
                    flip_left_angle,
                    right_angle,
                    flip_right_angle
                ])

            images = np.array(images, dtype = 'float32')
            angles = np.array(angles, dtype = 'float32')
            # print(images.shape, angles.shape)
            # exit()

            yield shuffle(images, angles)

def generate_validation_sample(samples, batch_size = BATCH_SIZE):
    sample_count = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, sample_count, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0].strip())
                center_image = transform_image(center_image)
                images.extend([center_image])

                center_angle = np.array(
                    batch_sample[3],
                    dtype = 'float32'
                )
                center_angle = transform_angle(center_angle)
                angles.extend([center_angle])

            images = np.array(images, dtype = 'float32')
            angles = np.array(angles, dtype = 'float32')

            yield shuffle(images, angles)

train_generator = generate_train_sample(train_set)
validation_generator = generate_validation_sample(validation_set)

# Transform data
def transform_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
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
# stride_size = 1
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
# model.add(Dropout(DROPOUT))
convolution_filter = 36
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid',
    subsample = (stride_size, stride_size)
))
# model.add(Dropout(DROPOUT))
convolution_filter = 48
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid',
    subsample = (stride_size, stride_size)
))
# model.add(Dropout(DROPOUT))
convolution_filter = 64
kernel_size = 3
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid'
))
# model.add(Dropout(DROPOUT))
model.add(Convolution2D(
    convolution_filter,
    kernel_size,
    kernel_size,
    border_mode = 'valid'
))
# model.add(Dropout(DROPOUT))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# model.summary()
# exit()

# Train model
adam = Adam(lr = LEARNING_RATE)
model.compile(optimizer = adam, loss = 'mse')
start_time = time.time()
history = model.fit_generator(
    train_generator,
    samples_per_epoch = samples_per_epoch,
    nb_epoch = EPOCH,
    verbose = VERBOSITY,
    validation_data = validation_generator,
    nb_val_samples = validation_samples
)
training_time = time.time() - start_time
samples_per_second = samples_per_epoch * EPOCH / training_time

# Show metric
model.summary()
print('Train set size:', len(train_set))
print('Batch size:', BATCH_SIZE)
print('Learning rate:', LEARNING_RATE)
print('Epoch:', EPOCH)
print('Training time: %i s' % training_time)
print('Samples per second: %i' % samples_per_second)

# Save model
model.save(MODEL_FILE)

# Plot loss
loss_chart = pyplot.figure()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.legend(['Training', 'Validation'])
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
loss_chart.savefig('loss.png')