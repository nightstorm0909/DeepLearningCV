
from nn.perceptron import Perceptron
import numpy as np 

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define the perceptron and train it
print('[INFO] training perceptron....')
p = Perceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs = 20)

# evaluate the perceptron
print('[INFO] testing the perceptron...')

# loop over the data points
for (x, target) in zip(X, y):
	# make the prediction on the data point and display the result
	pred = p.predict(x)
	print('[INFO] data = {}, ground-truth = {}, pred = {}'. format(x, target[0], pred))