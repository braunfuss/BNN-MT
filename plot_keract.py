from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Masking, Dense, Dropout, Activation, Conv2D, Flatten,  MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from pathlib import Path
import _pickle as pickle

import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf


from pyrocko.gf import LocalEngine
from pyrocko import moment_tensor as pmt
from pyrocko import gf, util
from pyrocko import model as pyrocko_model
import logging
logger = logging.getLogger('pyrocko.trace')
logger.setLevel(logging.ERROR)

import cnn_util
import plotting_functions as plf
import waveform_processing as wp
from mtqt_source import MTQTSource


import numpy as np
import os
pi = np.pi

np.random.seed(1234)
tf.random.set_seed(1234)

# Use the following store directory
store_dirs = "gf_stores/"

# Use the following store id

store_id="mojavelargemlhf"

engine = LocalEngine(store_superdirs=[store_dirs])


model = tf.keras.models.load_model('models/model_mechanism_single_gp_bnn_MT_mojavelargemlhf_5000.tf', compile=False) 
#model = tf.keras.models.load_model('/media/asteinbe/aki/models_halfwork/model_mojavelargemlhf_35.919999999999995_-117.68999999999997_4000', compile=False)
# model compilation in seperate step because we use Tensorflow probability layers and loss
model.compile(optimizer=Adam(), 
              loss=cnn_util.loss_function_negative_log_likelihood())

# To be able to evaluate the neg-log likelihood given certaint input we also need to load the weights again
#checkpoint_status = model.load_weights('models/model_weights_mechanism_single_gp_bnn_MT_mojavelargemlhf_5000')

data_dir="data_syn/events/"

#data_dir="data/events/"
# Retrieve the downloaded waveforms
store = engine.get_store(store_id)
waveforms_events, nsamples, events, waveforms_shifted = wp.load_data(data_dir, store_id, engine=engine)

data_events, nsamples = wp.prepare_waveforms(waveforms_events)
data_events = cnn_util.convert_data_events_to_input(data_events)


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

ax = plt.subplot(111)
im = ax.imshow(np.random.uniform(0., 1., 3600).reshape((1, 3600)), cmap="gray")
#d = np.linspace(-0.1, 1., 100, dtype="float").ravel()
#plt.imshow(d)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.05)

#plt.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.1, 1])

plt.show()



ax = plt.subplot(111)
im = ax.imshow(np.linspace(0., 1., 100, dtype="float").reshape((10, 10)), cmap="gray")

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05)

plt.colorbar(im, cax=cax, orientation="horizontal")

plt.show()

import keract


# Getting all the activations using keract
activations = keract.get_activations(model, data_events, layer_names=["Activation"])
#activations = keract.get_activations(model, data_events, layer_names=["activation_133"])
# Plotting using keract
keract.display_heatmaps(activations, data_events, cmap="gray",save=True, directory='picss') # to save the files use: save=True, directory='pics' f

#activations = keract.get_activations(model, data_events, layer_names=["conv2d_399", "conv2d_400", "conv2d_401", "max_pooling2d_133"])
activations = keract.get_activations(model, data_events, layer_names=["TimeConv0", "Dropout0", "TimeConv1", "Dropout1", "StationConv", "Dropout2", "Pooling", "DenseFlipout"])
## Plotting using keract
keract.display_heatmaps(activations, data_events, save=True, directory='picss', cmap="hot") # to save the files use: save=True, directory='pics' 




