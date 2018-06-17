# import the necessary packages
from nn.conv import LeNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture visualization graph to the disk

model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file = "lenet.png", show_shapes = True)
# plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')