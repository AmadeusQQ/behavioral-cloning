# Import libraries
from datetime import datetime
from flask import Flask
from io import BytesIO
from keras import __version__ as keras_version
from keras.models import load_model
from PIL import Image
import argparse
import base64
import cv2
import eventlet
import eventlet.wsgi
import h5py
import numpy as np
import os
import shutil
import socketio

# Set parameters
MIN_SPEED = 8
MAX_SPEED = 10

IMAGE_WIDTH = 160
IMAGE_LENGTH = 320
# IMAGE_DEPTH = 1
# IMAGE_WIDTH = 80
# IMAGE_LENGTH = 160
IMAGE_DEPTH = 3

model = None

sio = socketio.Server()
app = Flask(__name__)

def transform_image(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
    image = np.array(image, dtype = 'float32')

    return image.reshape(
        IMAGE_WIDTH,
        IMAGE_LENGTH,
        IMAGE_DEPTH
    )

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data["steering_angle"]
        throttle = data["throttle"]
        speed = data["speed"]
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = transform_image(image_array)
        steering_angle = float(
            model.predict(image_array[None, :, :, :], batch_size = 1)
        )
        
        if float(speed) < MIN_SPEED:
            throttle = 0.2
        elif float(speed) > MAX_SPEED:
            throttle = 0.0
        else:
            throttle = 0.1
        
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime(
                '%Y_%m_%d_%H_%M_%S_%f'
            )[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data = {}, skip_sid = True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data = {
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Remote Driving')
    parser.add_argument(
        'model',
        type = str,
        help = 'Path to model h5 file. ' +
            'Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to image folder. ' +
            'This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    f = h5py.File(args.model, mode = 'r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)
        
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    app = socketio.Middleware(sio, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)