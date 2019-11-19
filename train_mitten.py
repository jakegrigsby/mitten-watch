import os
import argparse

import tensorflow as tf

import data
from train_recognition import make_results_dirs, make_untrained_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='between_crabs')
    parser.add_argument('-train', type=int, default=25)
    parser.add_argument('-test', type=int, default=200)
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('-base', default=None)
    args = parser.parse_args()

    log_dir, ckpt_dir = make_results_dirs(args.name)

    if not args.base:
        model = make_untrained_model()
    else:
        model = tf.keras.models.load_model(os.path.join(args.base, 'checkpoints'))

    model.pop()

    model = tf.keras.Sequential([model, tf.keras.layers.Dense(7, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(.01))])

    metrics =[
            tf.keras.metrics.Precision(name='precision', class_id=1),
            tf.keras.metrics.Recall(name='recall', class_id=1),
            tf.keras.metrics.Accuracy(name='acc'),
            ]

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)

    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.ModelCheckpoint(ckpt_dir, save_freq=10000)
            ]

    train, val = data.load.open_crab(args.train, args.test, args.batch)
    print(f"Openness: {100*data.load.openness(args.train, args.test)}%")
    history = model.fit(train, validation_data=val, epochs=25, callbacks=callbacks, verbose=1)
