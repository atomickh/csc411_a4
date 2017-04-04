
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import pickle
import sys
from pylab import *
import matplotlib.pyplot as plt

snapshot = pickle.load(open("new_snapshot_weights_2.pkl", 'rb'))
weights = snapshot["weights"]

e = np.arange(1, len(weights)+1)

weights0 = np.array(weights)[:, :, 0]

colors = ['b', 'g', 'r', 'y']
for (i, color) in enumerate(colors):
    arr = weights0[:, i]
    plt.plot(e, arr, color, label="Weights "+str(i))

plt.grid()
plt.legend()
plt.show()
