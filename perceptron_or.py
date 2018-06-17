
from nn.perceptron import Perceptron
import numpy as np

# construct the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define our perceptron and train it
print('[INFO] training perceptron...')
p = Perceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs = 20)

# now we evaluate the perceptron
print('[INFO] testing perceptron...')

# LOOP OVER ALL DATA POINT
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result

	pred = p.predict(x)
	print('[INFO] data = {}, ground-truth = {}, pred = {}'.format(x, target[0], pred))