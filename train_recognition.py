import os
import argparse

import tensorflow as tf

def make_results_dirs(run_name, base_path='saves'):
    """ Create directory to save logs & checkpoints

    Creates new directory for a training run. Adds unique number
    to end of run_name if run_name has already been used.

    Args:
        run_name (str): Name of this experiment/training run.
        base_path (str, optional): name of root directory to put
             this run's folder in. Will be created if it doesn't already exist.
    
    Returns:
        log directory (str)
        checkpoint directory (str)
   """
    base_dir = os.path.join('saves/', run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir)
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir)
    return log_dir, checkpoint_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='run')
    args = parser.parse_args()

    model = tf.keras.models.Sequential([
        tf.keras.applications.MobileNetV2(include_top=False),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    log_dir, ckpt_dir = make_results_dirs(args.name)

    callbacks = [
            tf.keras.callbacks.Tensorboard(log_dir=log_dir),
            tf.keras.EarlyStopping(monitor='f1'),
            tf.keras.callbacks.ModelCheckpoint(ckpt_dir, monitor='f1', save_best_only=True)
            ]

    for negative_training_classes in range(5, 205, 25):
        pos_train, neg_train, pos_test, neg_test = data.imagenet_open_set(negative_training_classes)

