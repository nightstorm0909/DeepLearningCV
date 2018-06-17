from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
	# compute the sigmoid activation value for a given value
	return 1.0 / (1 + np.exp(-x))

def predict(X, W):
	# take the dot product between features and the weight matrix
	preds = sigmoid_activation(X.dot(W))

	# apply a step function to threshold the outputs to bianry class labels

	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1

	# return the prediction
	return preds

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type = float, default = 100, help = '# of epochs')

ap.add_argument('-a', '--alpha', type = float, default = 0.01, help = 'learning rate')
# vars([object]): Return the __dict__ attribute for a module, class, instance, or any other object with a __dict__ attribute
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points where each data points is a 2D faeture vector
# Generate isotropic Gaussian blobs for clustering
(X, y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1.5, random_state = 1)
#print('y.shape: ', y.shape, '\n')
y = y.reshape((y.shape[0], 1))	# numpy.reshape(a, newshape, order = 'C')
#print('y.shape: ', y.shape, '\n')

# insert a column of 1's as the last entry in the feature matrix for the bias trick

X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of the data for training and
# the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)
#print('trainX.shape: ', trainX.shape, 'trainY.shape: ',trainY.shape,'\n')

# initialize the weight matrix and list of losses
print('[INFO] training ...')
W = np.random.randn(X.shape[1], 1)
losses = []
#print('W.shape: ', W.shape, '\n')
# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):		# numpy.arange([start, ]stop, [step, ]dtype=None)
	# take the dot product between features 'X' and the weight matrix 'W',
	# then pass this value through the sigmoid activation function, thereby
	# giving us the predictions on the dataset
	preds = sigmoid_activation(trainX.dot(W))
	#print(trainY[0 : 10,:], '\n')
	# now that we have our prediction, we need to determine the 'error',
	# which is the difference between our predictions and the true values

	error = preds - trainY
	loss = np.sum(error ** 2)
	losses.append(loss)

	# the gradient descent update is the dot product between the features and the 
	# error of the predictions
	gradient = trainX.T.dot(error)

	# in the update stage, all we need to do is 'nudge' the weight matrix
	# in the negative direction of the gradient (hence the term 'gradient descent'
	# by taking a small stem towards a set of 'more optimal' parameters)

	W += -args['alpha'] * gradient

	# check to see if an update should be displayed
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print('[INFO] epoch = {}, loss = {:.7f}'.format(int(epoch + 1), loss))


# evaluate the model
print('[INFO] evaluating ...')
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use('ggplot')
plt.figure()
plt.title('data')
plt.scatter(testX[:, 0], testX[:, 1], marker = 'o', c = testY[:, 0], s = 30)

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()