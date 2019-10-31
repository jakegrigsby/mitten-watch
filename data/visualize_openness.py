
import numpy as np
import matplotlib.pyplot as plt

import load

def plot_binary_openness():
    training = [i for i in range(200)]
    testing = [i for i in range(200)]
    openness_scores = np.zeros((200, 200,))*np.nan
    for train in training:
        for test in testing:
            if test > train:
                openness_scores[test, train] = load.openness(train, test)
    plt.imshow(100*openness_scores, cmap='Blues_r', origin='lower')
    plt.xlabel('Training Classes', fontsize=15)
    plt.ylabel('Testing Classes', fontsize=15)
    plt.colorbar()
    plt.title('Openness (%)', fontsize=20)
    plt.show()


if __name__ == "__main__":
    plot_binary_openness()
