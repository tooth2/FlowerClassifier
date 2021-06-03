#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import logging
import argparse
import sys
import json
from PIL import Image
import glob
import os

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()
parser.add_argument ('--image_dir', default='./test_images/*.jpg', action='store', help = 'Path to image.', type = str)
parser.add_argument('--checkpoint', default = 'my_model.h5' , action="store", help='Point to checkpoint file as str.', type=str)
parser.add_argument ('--top_k', default = 5, action="store", help = 'Top K most likely classes.', type = int)
parser.add_argument ('--category_names' , default = 'label_map.json', action="store", help = 'Mapping of categories to real names.', type = str)

print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)

commands = parser.parse_args()
image_dir = commands.image_dir
saved_model = commands.checkpoint
classes = commands.category_names
top_k = commands.top_k

print(image_dir)
print(saved_model)
print(classes)
print(top_k)

# `custom_objects` tells keras how to load a `hub.KerasLayer`
reloaded = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer': hub.KerasLayer})
#reloaded = tf.keras.models.load_model(saved_model, compile = False)
reloaded.summary()
# Create the process_image function
with open(classes, 'r') as f:
    class_names = json.load(f)

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (224, 224))/255.0
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    print(image_path, "top probabilities:",top_values.numpy()[0])
    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]
    print(image_path, 'top classes:', top_classes)
    return top_values.numpy()[0], top_classes

files = glob.glob(image_dir)
for image_path in files:
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    probs, classes = predict(image_path, reloaded, top_k)
