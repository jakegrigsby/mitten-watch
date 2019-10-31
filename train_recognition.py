import os
import argparse

import tensorflow as tf
import tensorflow_hub as hub

import data

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
    parser.add_argument('-train', type=int, default=150)
    parser.add_argument('-test', type=int, default=200)
    parser.add_argument('-batch', type=int, default=64)
    args = parser.parse_args()



    log_dir, ckpt_dir = make_results_dirs(args.name)

    mobilenet = tf.keras.applications.MobileNetV2(input_shape=[data.load.IMG_HEIGHT, data.load.IMG_WIDTH, 3], include_top=False, weights='imagenet')
    mobilenet.trainable = False
    model = tf.keras.models.Sequential([
        mobilenet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.summary()

    metrics =[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            ]

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)

    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.EarlyStopping(monitor='precision'),
            tf.keras.callbacks.ModelCheckpoint(ckpt_dir, monitor='recall', save_best_only=True)
            ]

    train, val = data.load.open_set(args.train, args.test, args.batch)

    print(f"Openness: {100*data.load.openness(args.train, args.test)}%")
    history = model.fit(train, validation_data=val, epochs=100, callbacks=callbacks, verbose=1)
