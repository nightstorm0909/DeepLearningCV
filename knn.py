from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor

from imutils import paths
import argparse

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")

ap.add_argument("-k", "--neighbors", type = int, default = 1, help = "# of nearest neighbors for classification")

ap.add_argument("-j", "--jobs", type = int, default = -1, help = "# of jobs for k-NN distance (-1 uses all available cores)")
# vars([object]): Return the __dict__ attribute for a module, class, instance, or any other object with a __dict__ attribute
args = vars(ap.parse_args())	# parse_args: Convert argument strings to objects and assign them as attributes of the namespace

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initializa the image preproceso, load the dataset from the disk and reshape the data matrix

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
print('data.shape: ', data.shape, ' labels.shape: ', labels.shape, '\n')
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print('[INFO] features matrix: {: .1f}MB'.format(data.nbytes / (1024 * 1000.0)))
# ndarray.nbytes = Total bytes consumed by the elements of the array
# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of the data for training and remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)


# train and evaluate a k-NN classifier on the raw pixel intensities
print('[INFO]cevaluating k-NN classifier...')
model = KNeighborsClassifier(n_neighbors = args["neighbors"], n_jobs = args["jobs"])

model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names = le.classes_))
# sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
# returns Text summary of the precision, recall, F1 score for each class