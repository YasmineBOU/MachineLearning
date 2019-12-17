
import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import numpy
import random

from keras import backend
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten

# Only used for debug
from pprint import pprint


# Some needed constants
POSITIVE_DATASET_DIR_PATH = "dataset/TrainingAndValidating/Uninfected"  #'dataset/Uninfected'
NEGATIVE_DATASET_DIR_PATH = "dataset/TrainingAndValidating/Parasitized" # 'dataset/Parasitized'
IMAGE_WIDTH               = 32
IMAGE_HEIGHT              = 32
IMAGE_INPUT_SHAPE         = (0, 0, 0)
IMAGE_DIMENSIONS          = (IMAGE_WIDTH, IMAGE_HEIGHT)
NB_CHANNELS               = 3
CHANNELS_FIRST            = True  


# Method to set the input shape of images
def setImageInputShape():
    global IMAGE_INPUT_SHAPE

    if backend.image_data_format() == "channels_first":
        IMAGE_INPUT_SHAPE = (NB_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)

    else:
        IMAGE_INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, NB_CHANNELS)
        global CHANNELS_FIRST
        CHANNELS_FIRST = False    


def preprocessImage(imageFilePath):
    
    im = cv2.imread(imageFilePath)
    im = cv2.resize(im, IMAGE_DIMENSIONS)
    im = image.img_to_array(im)
    im = numpy.expand_dims(im, axis=0)

    return im


# Method that preprocesses images
# and returns an array of arrays
def preprocessImages(dirPath):
    imagesArray = []

    for imageName in sorted(os.listdir(dirPath)):
        if imageName.endswith("png"):
            imagePath = os.path.join(dirPath, imageName)
            # print("image: ", imagePath)
            imagesArray.append(preprocessImage(imagePath))

    return imagesArray


def splitData(X, y, testSize, randomState):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    return X_train, X_test, y_train, y_test 


# Method that creates a model
# and returns it
def createModel(nbClasses):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',input_shape=IMAGE_INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten()) # get a single vector from the MaxPooling2D operation
   
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='sigmoid'))

    return model
    

def compileModel(model, lossType, opt, metricsList):
    print("**********  Compile model  ********** \n")
    model.compile(loss=lossType, optimizer=opt, metrics=metricsList)


def amplifyData():
    return  ImageDataGenerator(
                                horizontal_flip=True,
                                rescale=0.25,
                                zoom_range=0.2,
                                width_shift_range=[-100, 100],

                                # vertical_flip=True,
                                # height_shift_range=[-100, 100],

            )


def fitModel(model, X_train, y_train, validationData, nbEpochs, batchSize):
    print("**********  Fit model  ********** \n")
    return model.fit(X_train, numpy.array(y_train), epochs=nbEpochs, batch_size=batchSize, verbose=2)



def fitModelGenerator(model, imageDataGenerator, X_train, y_train, validationData, nbEpochs, batchSize):
    print("**********  Fit model  ********** \n")
    model.fit_generator(imageDataGenerator.flow(
                                                X_train, 
                                                numpy.array(y_train), 
                                                batch_size=batchSize),
                                                epochs=nbEpochs, 
                                                validation_data=validationData,
    )


def printModelSummary(model):
    print("**********  Summary  **********\n")
    print(model.summary())    


def evaluateModel(model, XyTest):
    print("\n\n**********  Evaluate model  ********** \n")
    scores = model.evaluate(XyTest[0], XyTest[1])
    print("\tAccuracy: %.2f%%\n\n" % (scores[1]*100))





# Main program
if __name__ == "__main__":

    setImageInputShape()

    print("\n\nPreprocessing the images in the folder %s ...\n" % (POSITIVE_DATASET_DIR_PATH))
    X = preprocessImages(POSITIVE_DATASET_DIR_PATH)
    y = [0] * len(X)
    print("\n\nPreprocessing the images in the folder %s ...\n" % (NEGATIVE_DATASET_DIR_PATH))
    X += preprocessImages(NEGATIVE_DATASET_DIR_PATH)
    y += [1] * (len(X) - len(y))
    
    Xy = list(zip(X, y))
    random.shuffle(Xy)
    X, y = zip(*Xy)

    
    X = numpy.array(X, dtype="float") / 255.0
    y = numpy.array(y)
    

    if CHANNELS_FIRST:
        X = X.reshape(X.shape[0], NB_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        X = X.reshape(X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, NB_CHANNELS)

    X_train, X_test, y_train, y_test  = splitData(X, y, 0.25, 42)

    # y_train = y_train.reshape(-1, 1)

    model = createModel(1)
    compileModel(model, "binary_crossentropy", "adam", ["accuracy"])
    printModelSummary(model)
    # imageDataGenerator = amplifyData()
    # fitModelGenerator(model, imageDataGenerator, X_train, y_train, (numpy.array(X_test), y_test), nbEpochs=25, batchSize=10)
    epochs = 25 
    History = fitModel(model, X_train, y_train, (numpy.array(X_test), y_test), nbEpochs=epochs, batchSize=10)
    printModelSummary(model)
    evaluateModel(model, (X_test, y_test))

    model.save("basicModel.h5")
    model.save_weights("basicWeights.h5")
    plot_model(model, to_file='basicModelLayers.png', show_shapes=True, show_layer_names=True)


    plt.figure()
    # pprint(History.history)
    plt.plot(numpy.arange(0, epochs), History.history["loss"], label="train_loss")
    plt.plot(numpy.arange(0, epochs), History.history["acc"], label="train_acc")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig("graph.png")











