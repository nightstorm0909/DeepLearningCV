# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be 'channel list'

		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using 'channel first', update the input shape
		if K.image_data_format() == 'channels_first':
			inputShape = (depth, height, width)

		# define the first (and only) CONV => RELU layer
		model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = inputShape))
		# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, 
		# use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
		# activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

		model.add(Activation('relu'))

		# softmax classifier
		model.add(Flatten())	  # keras.layers.Flatten(data_format=None)
		model.add(Dense(classes)) # keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
		# 							bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
		# 							kernel_constraint=None, bias_constraint=None)
		model.add(Activation('softmax')) # keras.layers.Activation(activation)

		# return the constructed network architecture
		return model