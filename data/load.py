import json
import os
import glob
import random
import math

import tensorflow as tf

IMG_WIDTH = 160
IMG_HEIGHT = 160

def _rel_path(path):
    rel_path = ''.join(os.path.split(__file__)[:-1])
    return os.path.join(rel_path, path)

def _load_imagenet_label_map():
    with open(_rel_path('tiny-imagenet/imagenet_class_index.json')) as fp:
        num_to_name_json = json.load(fp)
        id_to_name = {}
        for num in num_to_name_json.keys():
            id_, name  = num_to_name_json[num]
            id_to_name[id_] = name
    return id_to_name

ID_TO_NAME = _load_imagenet_label_map()

def _get_label_from_folder_name(class_dir):
    id_ = os.path.split(class_dir)[-1]
    return ID_TO_NAME[id_]

def _load_train():
    path = _rel_path(os.path.join('tiny-imagenet', 'train'))
    filenames = []
    labels = []
    for class_dir in glob.glob(path + '/*'):
        label = _get_label_from_folder_name(class_dir)
        img_filenames = glob.glob(class_dir + '/images/*')
        filenames.extend(img_filenames)
        labels.extend([label]*len(img_filenames))
    return filenames, labels

def _load_crabs(train_split=.7):
    filenames = []
    for filename in glob.glob(_rel_path('cleaned_crabs/*')):
        filenames.append(filename)
    random.shuffle(filenames)
    split_idx = math.floor(len(filenames)*train_split)
    train_filenames = filenames[:split_idx]
    val_filenames = filenames[split_idx:]
    train_labels = ['crab']*len(train_filenames)
    val_labels = ['crab']*len(val_filenames)
    return (train_filenames, train_labels), (val_filenames, val_labels)

def _load_crab_classes(train_split=.7):

    def gather_class(folder, label):
        filenames = [filename for filename in glob.glob(f"{folder}/*")]
        random.shuffle(filenames)
        split_idx = math.floor(len(filenames)*train_split)
        train_filenames = filenames[:split_idx]
        val_filenames = filenames[split_idx:]
        train_labels = [label]*len(train_filenames)
        val_labels = [label]*len(val_filenames)
        return (train_filenames, train_labels), (val_filenames, val_labels)

    (bluecrab_t_filenames, bluecrab_t_labels), (bluecrab_v_filenames, bluecrab_v_labels) = gather_class(_rel_path('Crab-Negatives/BlueCrabPhotos'), 'bluecrab')
    (ghostcrab_t_filenames, ghostcrab_t_labels), (ghostcrab_v_filenames, ghostcrab_v_labels) = gather_class(_rel_path('Crab-Negatives/GhostCrabPhotos'), 'ghostcrab')
    (horseshoecrab_t_filenames, horseshoecrab_t_labels), (horseshoecrab_v_filenames, horseshoecrab_v_labels) = gather_class(_rel_path('Crab-Negatives/HorseshoeCrabPhotos'), 'horseshoecrab')
    (cmc_t_filenames, cmc_t_labels), (cmc_v_filenames, cmc_v_labels) = gather_class(_rel_path('Crab-Negatives/CMCPhotos'), 'cmc')
    (hermitcrab_t_filenames, hermitcrab_t_labels), (hermitcrab_v_filenames, hermitcrab_v_labels) = gather_class(_rel_path('Crab-Negatives/HermitCrabPhotos'), 'hermitcrab')
    (redcrab_t_filenames, redcrab_t_labels), (redcrab_v_filenames, redcrab_v_labels) = gather_class(_rel_path('Crab-Negatives/RedCrabPhotos'), 'redcrab')

    all_t_filenames = bluecrab_t_filenames + ghostcrab_t_filenames + cmc_t_filenames + hermitcrab_t_filenames + redcrab_t_filenames + horseshoecrab_t_filenames
    all_v_filenames = bluecrab_v_filenames + ghostcrab_v_filenames + cmc_v_filenames + hermitcrab_v_filenames + redcrab_v_filenames + horseshoecrab_v_filenames
    all_t_labels = bluecrab_t_labels + ghostcrab_t_labels + cmc_t_labels + hermitcrab_t_labels + redcrab_t_labels + horseshoecrab_t_labels
    all_v_labels = bluecrab_v_labels + ghostcrab_v_labels + cmc_v_labels + hermitcrab_v_labels + redcrab_v_labels + horseshoecrab_v_labels

    return (all_t_filenames, all_t_labels), (all_v_filenames, all_v_labels)

def _load_val():
    path = _rel_path(os.path.join('tiny-imagenet', 'val'))
    filenames = []
    labels = []
    with open(_rel_path(os.path.join(path, 'val_annotations.txt'))) as f:
        for line in f.readlines():
            split_line = line.split('\t')
            filenames.append(_rel_path(os.path.join(path, 'images', split_line[0])))
            labels.append(_get_label_from_folder_name(split_line[1]))
    return filenames, labels

def _load_image(filename, label):
    image_str = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_str, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_HEIGHT, IMG_WIDTH])
    return image, label

def _make_dataset(filenames, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(_load_image)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def _make_final_dataset(filenames, labels, batch_size):
    labels = tf.one_hot(labels, 7)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(_load_image)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def openness_scheirer(training, testing, target=1):
    """Openness according to scheirer et al. Can't figure out
    why this doesn't work"""
    return 1. - math.sqrt(
            2*training/(testing+target)
            )

def openness(training, testing, target=1):
    """Intuitive openness
    training (int) : number of negative classes present during training.
    testing (int) : number of negative classses present during testing.
    target (int) : number of positive classes we care about classifying.
    """
    return 1 - (training+target)/(testing+target)

def open_set(train, test, batch_size):
    """Create an open set dataset.
    """
    assert test <= 200, "There are only 200 classes in Tiny Imagenet"
    assert train <= test, "Testing classes are a superset of Training"
    all_train_filenames, all_train_labels = _load_train()
    all_val_filenames, all_val_labels = _load_val()

    unique_classes = list(set(all_train_labels))
    random.shuffle(unique_classes)
    train_classes = set(unique_classes[:train])
    test_classes = set(unique_classes[:test])
    assert train_classes <= test_classes

    # get rid of unused classes
    for idx, label in enumerate(all_train_labels):
        if not label in train_classes:
            all_train_filenames[idx] = None
            all_train_labels[idx] = None
    train_filenames = [filename for filename in all_train_filenames if filename]
    train_labels = [label for label in all_train_labels if label]
    for idx, label in enumerate(all_val_labels):
        if not label in test_classes:
            all_val_filenames[idx] = None
            all_val_labels[idx] = None
    val_filenames = [filename for filename in all_val_filenames if filename]
    val_labels = [label for label in all_val_labels if label]

    (crab_t_files, crab_t_labels), (crab_v_files, crab_v_labels) = _load_crabs()
    train_filenames.extend(crab_t_files)
    train_labels.extend(crab_t_labels)
    val_filenames.extend(crab_v_files)
    val_labels.extend(crab_v_labels)

    train_labels = _convert_to_binary(train_labels)
    val_labels = _convert_to_binary(val_labels)

    train_dataset = _make_dataset(train_filenames, train_labels, batch_size)
    val_dataset = _make_dataset(val_filenames, val_labels, batch_size)
    return train_dataset, val_dataset

def open_crab(train, test, batch_size):
    assert test <= 200, "There are only 200 classes in Tiny Imagenet"
    assert train <= test, "Testing classes are a superset of Training"
    ####################################
    ## LOAD IMAGENET OPEN SET CLASSES ##
    ####################################
    all_train_filenames, all_train_labels = _load_train()
    all_val_filenames, all_val_labels = _load_val()
    unique_classes = list(set(all_train_labels))
    random.shuffle(unique_classes)
    train_classes = set(unique_classes[:train])
    test_classes = set(unique_classes[:test])
    assert train_classes <= test_classes
    # get rid of unused classes
    for idx, label in enumerate(all_train_labels):
        if not label in train_classes:
            all_train_filenames[idx] = None
            all_train_labels[idx] = None
    train_filenames = [filename for filename in all_train_filenames if filename]
    train_labels = [label for label in all_train_labels if label]
    for idx, label in enumerate(all_val_labels):
        if not label in test_classes:
            all_val_filenames[idx] = None
            all_val_labels[idx] = None
    val_filenames = [filename for filename in all_val_filenames if filename]
    val_labels = [label for label in all_val_labels if label]

    #######################
    ## LOAD CRAB CLASSES ##
    #######################
    (crab_t_filenames, crab_t_labels), (crab_v_filenames, crab_v_labels) = _load_crab_classes()

    def class_labels_to_int(labels):
        for idx, label in enumerate(labels):
            if label == 'cmc':
                labels[idx] = 1
            elif label == 'bluecrab':
                labels[idx] = 2
            elif label == 'ghostcrab':
                labels[idx] = 3
            elif label == 'horseshoecrab':
                labels[idx] = 4
            elif label == 'hermitcrab':
                labels[idx] = 5
            elif label == 'redcrab':
                labels[idx] = 6
            else:
                labels[idx] = 0
        return labels

    all_t_filenames = train_filenames + crab_t_filenames
    all_t_labels = train_labels + crab_t_labels

    all_v_filenames = val_filenames + crab_v_filenames
    all_v_labels = val_labels + crab_v_labels

    class_labels_to_int(all_t_labels)
    class_labels_to_int(all_v_labels)

    train_dset = _make_final_dataset(all_t_filenames, all_t_labels, batch_size)
    val_dset = _make_final_dataset(all_v_filenames, all_v_labels, batch_size)

    return train_dset, val_dset

def _convert_to_binary(labels):
    for idx, label in enumerate(labels):
        if label == 'crab':
            labels[idx] = 1
        else:
            labels[idx] = 0
    return labels

if __name__ == "__main__":
    train, val = open_set(50, 100, 32)

