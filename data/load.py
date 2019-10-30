import json
import os
import glob

import tensorflow as tf

IMG_WIDTH = 200
IMG_HEIGHT = 200

def load_imagenet_label_map():
    with open('tiny-imagenet-crabs/imagenet_class_index.json') as fp:
        num_to_name_json = json.load(fp)
        id_to_name = {}
        for num in num_to_name_json.keys():
            id_, name  = num_to_name_json[num]
            id_to_name[id_] = name
    return id_to_name

ID_TO_NAME = load_imagenet_label_map()

def get_label_from_folder_name(class_dir):
    id_ = os.path.split(class_dir)[-1]
    return ID_TO_NAME[id_]

def load_train():
    path = os.path.join('tiny-imagenet-crabs', 'train')
    filenames = []
    labels = []
    for class_dir in glob.glob(path + '/*'):
        label = get_label_from_folder_name(class_dir)
        img_filenames = glob.glob(class_dir + '/images/*')
        filenames.extend(img_filenames)
        labels.extend([label]*len(img_filenames))
    return filenames, labels

def load_val():
    path = os.path.join('tiny-imagenet-crabs', 'val')
    filenames = []
    labels = []
    with open(os.path.join(path, 'val_annotations.txt')) as f:
        for line in f.readlines():
            split_line = line.split('\t')
            filenames.append(os.path.join(path, 'images', split_line[0]))
            labels.append(get_label_from_folder_name(split_line[1]))
    return filenames, labels

def load_image(filename, label):
    image_str = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_str, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_HEIGHT, IMG_WIDTH])
    return image, label

def make_dataset(filenames, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(load_image)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset



if __name__ == "__main__":
    train = make_dataset(*load_train(), 32)
    val = make_dataset(*load_val(), 32)

