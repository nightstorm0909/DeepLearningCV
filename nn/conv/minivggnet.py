# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K

class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be 'channels last' and
		# the chaneels dimension itself
		model = Sequential()
		inputShape = (width, height, depth)
		chanDim = -1

		# if we are using 'channel first', update the input shape

		if K.image_data_format() == 'channels_first':
			inputShape = (depth, width, height)
			chanDim = 1

		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding = 'same',
			input_shape = inputShape))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis = chanDim))
		# keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
		# moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
		# gamma_constraint=None)
		model.add(Conv2D(32, (3, 3), padding = 'same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (2, 2)))
		model.add(Dropout(0.25))
		# keras.layers.Dropout(rate, noise_shape=None, seed=None): Dropout consists in randomly setting a fraction rate of input units to 0 at 
		# each update during training time, which helps prevent overfitting.
		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding = 'same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis = chanDim))
		model.add(Conv2D(64, (3, 3), padding = 'same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (2, 2)))
		model.add(Dropout(0.25))

		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation('softmax'))

		# return the constructed network
		return model