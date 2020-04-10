# this file creates the visualization of the model architecture

from keras.models import load_model
from keras.utils.vis_utils import plot_model

modelname = 'model.h5'

model = load_model(modelname)
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)