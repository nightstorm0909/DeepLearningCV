import numpy as np 
import cv2

#initialize the class labels and set the seed of the pseudorandom number generator so we can reproduce our result
labels = ['dog', 'cat', 'panda']
np.random.seed(1)

# randomly initialize the weight matrix and bias vector
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# load the example image, resize it and then flatten it to the feature vector representation
orig = cv2.imread('beagle.png')
image = cv2.resize(orig, (32, 32)).flatten()
# ndarray.flatten(order='C'): Return a copy of the array collapsed into one dimension. C gives row major

# compute the output scores by taking the dot product between the weight matrix and image pixels followed by adding in the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
	# zip(): Returns an iterator of tuples, where the i-th tuple contains the i-th element from each of the argument sequences 
	# or iterables.
	print('[INFO] {}: {:.2f}'.format(label, score))

# draw the label with the highest score on the image as prediction
cv2.putText(orig, 'Label: {}'.format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# img =	cv.putText(	img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]	)
# numpy.argmax(a, axis=None, out=None) : Returns the indices of the maximum values along an axis

# display the input image
cv2.imshow('image', orig)
cv2.waitKey(0)