import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
	def __init__(self, layers, alpha = 0.1):
		# initialize the list of weights matrices then store the network architecture
		# and learning rate
		self.W = []
		self.layers = layers	# A list of integers which represents the actual architecture of the feedforward network
		self.alpha = alpha
		self.losses = []
		# start looping from the index of the first layer but stop before we reach
		# the last two layers
		for i in np.arange(0, len(layers) - 2):
			# randomly initialize a weight matrix connecting the number of nodes in each
			# respective layer together, adding an extra node for the bias

			w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
			self.W.append(w / np.sqrt(layers[i]))	# This is done to normalize the variance of each neuron's output

		# the last two layers are a special case where the input connections need
		# a bias term but the output does not
		w = np.random.rand(layers[-2] + 1, layers[-1])
		self.W.append(w / np.sqrt(layers[-2]))

	def __repr__(self):
		# construct and return a string that represents the network architecture
		# This is a Python 'magic method', useful for debugging
		return 'Neural Network: {}'.format('-'.join(str(l) for l in self.layers))

	
	def sigmoid(self, x):
		# compute and return the sigmoid activation value for a given input

		return 1.0 / (1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		# compute the derivative of the sigmoid function assuming that 'x' has already been
		# passed through the sigmoid function

		return x * (1 - x)

	def fit(self, X, y, epochs = 1000, displayUpdate = 100):
		# insert a column of 1s as the last entry in the feature matrix for the
		# bias trick

		X = np.c_[X, np.ones((X.shape[0]))]
		
		# loop over the desired number of  epochs
		for epoch in np.arange(0, epochs):
			# loop over each individual data point and train the network on it
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)

			# check to see if we should display the training update
			if epoch == 0 or (epoch + 1 ) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				print('[INFO] epoch = {}, loss = {:.7f}'.format(epoch + 1, loss))
			self.losses.append(self.calculate_loss(X, y))


	def fit_partial(self, x, y):
		# construct our list of output activations for each layer as our data point flows 
		# through the network; the first activation is a special case -- it's just the input 
		# feature vector itself
		# x: an individual data point from design matrix;		y: the corresponding class label
		A = [np.atleast_2d(x)]		# np.atleast_2d: View inputs as arrays with at least two dimensions
		#print('fit partial:::::\nA: ', A, '\n')
		# FEEDFORWARD
		# loop over the layers in the network
		for layer in np.arange(0, len(self.W)):	# len(np.array) : returns the first dimension number
			# feedforward the activation at the current layer by taking the dot product
			# between the activation and the weight matrix -- this is called the 'net input'
			# to the current layer

			net = A[layer].dot(self.W[layer])

			# computing the 'net output' is simply applying our non linear activation
			# function to the net input
			out = self.sigmoid(net)

			# once we have the net output, add it to the list of activations

			A.append(out)
		#print('A: ', A, '\n')
		# BACKPROPAGATION
		# the first phase of the backpropagation is to compute the difference between
		# our prediction (the final output activation in the activation list) and the
		# true target value

		error = ( A[-1] - y )
		#print('error: ', error, '\n')
		# from here, we need to apply the chain rule and build the list of deltas 'D';
		# the first entry in the deltas is simply the error of the output layer times
		# the derivative of our activation function for the output value

		D = [error * self.sigmoid_deriv(A[-1])]
		#print('D: ', D, '\n')
		# once you understand the chain rule it becomes super easy to implement with a
		# for loop -- simply loop over the layers in reverse order (ignoring the last two
		# since we already have taken them into account)

		for layer in np.arange(len(A) - 2, 0, -1):
			# the delta for the current layer is equal to the delta of the previous layer
			# dotted with the weight matrix of the current layer followed by multiplying
			# the delta by the derivatives of the non linear activation function for the
			# for the activations of the current layers

			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

			# since we loop over the layers in reverse order we need to reverse the deltas
		#print('D: ', D, '\n')
		D = D[::-1]
		#print('D: ', D, '\n')
		# WEIGHT UPDATE PHASE
		# loop over the layers
		for layer in np.arange(0, len(self.W)):
			# update the weights by taking the dot product of the layer activations with
			# their respective deltas then multiplying this value by small learning rate
			# and adding in our weight matrix -- this is where the actual learning
			# takes place

			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

	def predict(self, X, addBias = True):
		# initialize the output prediction as the input feature -- this value will be forward
		# propagated through the network to obtain the final prediction

		p = np.atleast_2d(X)

		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1s as the lst entry in the feature matrix (bias)

			p = np.c_[p, np.ones((p.shape[0]))]

		# loop over our layers in the network
		for layer in np.arange(0, len(self.W)):
			# computing the output prediction is as simple as taking the dot product
			# between the current activation value 'p' and the weight matrix 
			# associated with the current layer, then passing this value through a 
			# non linear activation function

			p = self.sigmoid(np.dot(p, self.W[layer]))

		# return the predicted value
		return p

	def calculate_loss(self, X, targets):
		# make the predictions for the input data points then compute the loss

		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias = False)
		loss = 0.5 * np.sum((predictions - targets) ** 2)
		#print('loss: ', loss, '\n')
		# return the loss
		return loss

	def plot_loss(self):
		# plotting losses
		#print(len(self.losses))
		plt.plot(self.losses)
		plt.xlabel('losses')
		plt.ylabel('epoch')
		plt.show()