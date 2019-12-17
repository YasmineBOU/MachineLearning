
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import cv2
import numpy
import pickle
# Python libraries
import random

from keras import backend
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten

# Only used for debug
from pprint import pprint

# Local library
from program import preprocessImage

# Some useful constants
CMD_NB_ARGUMENTS = 2
PREDICTIONS_DATASET_DIR_PATH = "dataset/TestPredictions" 

# CMD_NB_ARGUMENTS = 4

def makePrediction(imageFilePath):
	img = preprocessImage(imageFilePath)
	res = model.predict(img)
	print("\n\n{}\t=======>\tUninfected cell".format(imageFilePath)) if res[0][0] == 0 else print("\n\n{}\t=======>\tParasitized cell\n".format(imageFilePath)) #else print("Some error(s) has occured\n\n")	
	
	return "Uninfected" if res[0][0] == 0 else "Parasitized"

if __name__ == "__main__":


	if len(sys.argv) < CMD_NB_ARGUMENTS:
		print("\n\nUse:\n\npython3 predict.py [basicModel|ResNet50Model]\nOr:\n" 
			  "python3 predict.py [basicModel|ResNet50Model] file_path_to_image(s)\n\n")
		exit(1)

	model  = ""


	if sys.argv[1].lower() == "basicmodel":
		print("Loading the basic model (basicModel.h)\n")
		model = load_model("basicModel.h5")
		model.load_weights("basicWeights.h5")

	else:
		print("Loading the ResNet 50 model (ResNet50Model.h5)\n")
		model = load_model("ResNet50Model.h5")
		model.load_weights("ResNet50Weights.h5")



	if len(sys.argv) > CMD_NB_ARGUMENTS:
		imagesFilePaths = [sys.argv[i] for i in range(CMD_NB_ARGUMENTS - 1, len(sys.argv))]

		model = load_model(sys.argv[1])
		model.load_weights(sys.argv[2])
		
		for imageFilePath in imagesFilePaths:
			makePrediction(imageFilePath)

	else:	
		ratePredictions = []	
		for folderName in os.listdir(PREDICTIONS_DATASET_DIR_PATH):
			folder = os.path.join(PREDICTIONS_DATASET_DIR_PATH, folderName)
			nbImages, nbPredictedImages = 0, 0
			for imageFilename in os.listdir(folder):
				imageFilePath = os.path.join(folder, imageFilename)

				nbPredictedImages += 1 if makePrediction(imageFilePath) == folderName else 0
				nbImages += 1

			ratePredictions.append((folder, float(nbPredictedImages / nbImages) * 100))

		print("\n\n#################################################################################################\n\n")
		for prediction in ratePredictions:

			print("Prediction with the folder {}: {}%".format(prediction[0], prediction[1]))

		print("\n\n")