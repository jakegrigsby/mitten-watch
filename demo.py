import argparse
import os

import tensorflow as tf
import matplotlib.pyplot as plt

import data

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('-run', type=str)
     parser.add_argument('-test', type=int, default=10)
     args = parser.parse_args()

     _, val = data.load.open_set(train=3, test=args.test, batch_size=16)
     imgs, _ = val.__iter__().__next__()
     model = tf.keras.models.load_model(os.path.join(args.run, 'checkpoints'))
     preds = model(imgs)
     fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(5,5))
     idx = 0
     for row in range(4):
         for col in range(4):
             axs[row, col].imshow(imgs[idx])
             axs[row, col].set_title("Crab" if preds[idx].numpy() >= .5 else "Not Crab")
             axs[row, col].set_xticks([])
             axs[row, col].set_yticks([])
             idx += 1
     plt.show()
            




