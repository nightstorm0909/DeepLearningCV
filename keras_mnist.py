from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 
import argparse


# Contruct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required = True, help = 'path to the out loss/accuracy plot')

args = vars(ap.parse_args())

# grad the MNIST dataset (if this is the first time running this script, the download may
# take a minute -- the 55mb MNIST dataset will be downloaded)

print('[INFO] loading MNIST (full) dataset.....')
dataset = datasets.fetch_mldata('MNIST Original')

# scale the raw pixel intensities to the range [0, 1.0], then construct the training
# and testing splits
data = dataset.data.astype('float') / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size = 0.25)


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape = (784,), activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(10, activation = 'softmax'))

# train the model using SGD
print('[INFO] training network....')
sgd = SGD(0.01)			# keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])	# Configures the model for training
# compile(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 100, batch_size = 128)
#fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, 
#    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size = 128)		# predict(self, x, batch_size=None, verbose=0, steps=None)
print(classification_report(testY.argmax(axis = 1), 
	predictions.argmax(axis = 1),
	target_names = [str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label = 'val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['output'])