import argparse
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import data

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('-run', type=str)
     args = parser.parse_args()

     train, val = data.load.open_crab(train=0, test=0, batch_size=16)
     imgs, _ = val.__iter__().__next__()
     model = tf.keras.models.load_model(os.path.join(args.run, 'checkpoints'))
     preds = model(imgs)
     fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(5,5))
     idx = 0
     idx_to_name = {0:"Other", 1:"Mitten", 2:"Blue", 3:"Ghost", 4:"Horseshoe", 5:"Hermit", 6:"Red"}
     for row in range(4):
         for col in range(4):
             axs[row, col].imshow(imgs[idx])
             axs[row, col].set_title(idx_to_name[np.argmax(preds[idx])])
             axs[row, col].set_xticks([])
             axs[row, col].set_yticks([])
             idx += 1
     plt.show()
            




