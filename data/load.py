""" Build an Image Dataset in TensorFlow.

For this example, you need to make your own set of images (JPEG).
We will show 2 different ways to build that dataset:

- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |             
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```

Below, there are some parameters that you need to change (Marked 'CHANGE HERE'), 
such as the dataset path.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
import os

import tensorflow as tf
from PIL import Image

DATASET_BASE_PATH = 'tiny-imagenet-crabs' # the dataset file or root folder path.
# Image Parameters
N_CLASSES = 201 # CHANGE HERE, total number of classes
IMG_HEIGHT = 200 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 200 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale


def load_dataset(dset_subdirectory):
    DATASET_BASE_PATH = os.path.join(DATASET_BASE_PATH, dset_subdirectory)
        # Reading the dataset
    # 2 modes: 'file' or 'folder'
    def read_images(dataset_path, mode, batch_size):
        imagepaths, labels = list(), list()
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg') or smaple.endswith('.JPEG') or sample.endswith('.JPG'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1

        # Convert to Tensor
        imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        # Build a TF Queue, shuffle data
        image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                     shuffle=True)

        # Read images from disk
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=CHANNELS)

        # Resize images to a common size
        image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

        # Normalize
        image = image * 1.0/127.5 - 1.0

        # Create batches
        X, Y = tf.train.batch([image, label], batch_size=batch_size,
                              capacity=batch_size * 8,
                              num_threads=4)

        return X, Y

