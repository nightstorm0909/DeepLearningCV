from nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# load the mnist dataset and apply min/max scaling to scale the pixel intensity
# values to the range [0, 1] (each image is represented by a 8 x 8 = 64-dim
# feature vector)
print('[INFO] loading MNIST (sample) dataset...')
digits = datasets.load_digits()		# If True, returns (data, target) instead of a Bunch object
data = digits.data.astype('float')	# data numpy ndarray
data = (data - data.min()) / (data.max() - data.min())	# min(): Return the minimum of an array.
print('[INFO] samples: {}, dim: {}'.format(data.shape[0], data.shape[1]))


# construct the training and testing split
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size = 0.25)


# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)	# Binarize labels in a one-vs-all fashion
testY = LabelBinarizer().fit_transform(testY)

# train the network
print('[INFO] training network....')
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print('[INFO] {}'.format(nn))
nn.fit(trainX, trainY, epochs = 1000)

# evaluate the network
print('[INFO] evaluating the network...')
predictions = nn.predict(testX)
predictions = predictions.argmax(axis = 1)
print(classification_report(testY.argmax(axis = 1), predictions))

nn.plot_loss()