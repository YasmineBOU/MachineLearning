
import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import numpy
import pickle
import random

import missinglink
from keras import backend
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D


# Only used for debug
from pprint import pprint


# Some needed constants
# TRAINING_DATASET_DIR_PATH = "dataset/TrainingAndValidating"
# TRAINING_DATASET_DIR_PATH = "dataset/TrainingAndValidating"
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
        print("In setImageInputShape():\t1)- IMAGE_INPUT_SHAPE: ", IMAGE_INPUT_SHAPE)

    else:
        IMAGE_INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, NB_CHANNELS)
        print("In setImageInputShape():\t2)- IMAGE_INPUT_SHAPE: ", IMAGE_INPUT_SHAPE)
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
def createModelOld(nbClasses):
    print("In createModel()\tIMAGE_INPUT_SHAPE: ", IMAGE_INPUT_SHAPE)
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
    


def createModel(nbClasses):

    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg'))
    model.add(Dense(nbClasses, activation="sigmoid"))

    pprint(model.layers)
    model.layers[0].trainable = False

    return model



def compileModel(model, lossType, opt, metricsList):
    print("**********  Compile model  ********** \n")
    model.compile(loss=lossType, optimizer=opt, metrics=metricsList)


def amplifyData():
    return ImageDataGenerator(
	                            horizontal_flip=True,
	                            rescale=0.25,
	                            zoom_range=0.2,
	                            width_shift_range=[-100, 100],
	
	                            # vertical_flip=True,
	                            # height_shift_range=[-100, 100],


           )


def fitModel(model, X_train, y_train, validationData, nbEpochs, batchSize):
    print("**********  Fit model  ********** \n")
    model.fit(X_train, numpy.array(y_train), epochs=nbEpochs, batch_size=batchSize, verbose=1)



def fitModelGenerator(model, imageDataGenerator, X_train, y_train, validationData, nbEpochs, batchSize):
    print("**********  Fit model  ********** \n")
    model.fit_generator(imageDataGenerator.flow(X_train, 
                                                numpy.array(y_train), 
                                                epochs=nbEpochs, 
                                                batch_size=batchSize, 
                                                verbose=2)
    )


def printModelSummary(model):
    print("**********  Summary  **********\n")
    print(model.summary())    


def evaluateModel(model, XyTest):
    print("\n\n**********  Evaluate model  ********** \n")
    scores = model.evaluate(XyTest[0], XyTest[1])
    print("Scores: ", scores)
    print("\tAccuracy: %.2f%%\n\n" % (scores[1]*100))





# Main program
if __name__ == "__main__":

    setImageInputShape()

    print("\n\nPreprocessing the images in the folder \'%s\' ...\n" % (POSITIVE_DATASET_DIR_PATH))
    X = preprocessImages(POSITIVE_DATASET_DIR_PATH)
    y = [0] * len(X)
    print("\n\nPreprocessing the images in the folder \'%s\' ...\n" % (NEGATIVE_DATASET_DIR_PATH))
    X += preprocessImages(NEGATIVE_DATASET_DIR_PATH)
    y += [1] * (len(X) - len(y))
    
    Xy = list(zip(X, y))
    random.shuffle(Xy)
    X, y = zip(*Xy)


    
    X = numpy.array(X, dtype="float") / 255.0
    y = numpy.array(y)
    print("type X: {}\tlen: {}\tshape: {}".format(type(X), len(X), X.shape))
    print("type y: {}\tlen: {}\tshape: {}".format(type(y), len(y), y.shape))
    print("CHANNELS_FIRST: ", CHANNELS_FIRST)

    if CHANNELS_FIRST:
        X = X.reshape(X.shape[0], NB_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        X = X.reshape(X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, NB_CHANNELS)

        
    # pprint(X)
    print("\n*************\nTHE NEW ONE\n*************\n")
    # pprint(y)

    print("type X: {}\tlen: {}\tshape: {}".format(type(X), len(X), X.shape))

    X_train, X_test, y_train, y_test  = splitData(X, y, 0.25, 42)

    # y_train = y_train.reshape(-1, 1)

    model = createModel(1)
    compileModel(model, "binary_crossentropy", "sgd", ["accuracy"])
    printModelSummary(model)


    missinglink_callback = missinglink.KerasCallback()

    # dataGenerator = ImageDataGenerator(prepocessing_function=prepocess_input)

    # trainGenerator = dataGenerator.flow_from_directory(

    #                 )


    fitModel(model, X_train, y_train, (numpy.array(X_test), y_test), nbEpochs=25, batchSize=10)
    printModelSummary(model)
    evaluateModel(model, (X_test, y_test))

    model.save("ResNet50Model.h5")
    model.save_weights("ResNet50Weights.h5")























# def createModel(nbClasses):
# 	imageShape = Input(shape=IMAGE_INPUT_SHAPE)
# 	base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=imageShape)
# 	base_model.summary()

# 	x = base_model.output
# 	x = GlobalAveragePooling2D()(x)
# 	# let's add a fully-connected layer
# 	x = Dense(64, activation='relu')(x)
# 	# and a logistic layer -- let's say we have 200 classes
# 	predictions = Dense(1, activation='sigmoid')(x)

# 	# this is the model we will train
# 	model = Model(inputs=base_model.input, outputs=predictions)

# 	# first: train only the top layers (which were randomly initialized)
# 	# i.e. freeze all convolutional InceptionV3 layers
# 	# for layer in model.layers:
# 	#     layer.trainable = False


# 	return model

# 	# last_layer = model.get_layer('avg_pool').output
# 	# x = Flatten(name='flatten')(last_layer)
# 	# out = Dense(1, activation='sigmoid', name='output_layer')(x)
# 	# resNetModel = Model(inputs=IMAGE_INPUT_SHAPE, outputs=out)
# 	# resNetModel.summary()


# 	# for layer in resNetModel.layers[:-1]:
# 	# 	layer.trainable = False

# 	# return resNetModel





# def createModel(nbClasses):

#     modelResNet = ResNet50(weights='imagenet',include_top=False)
#     modelResNet.summary()
#     last_layer = modelResNet.output
#     # add a global spatial average pooling layer
#     model = GlobalMaxPooling2D()(last_layer)
#     # model = Conv2D(32, (3, 3), padding='same', activation='relu')(model)
#     # model = MaxPooling2D(pool_size=(2, 2))(model)

#     # model = Conv2D(32, (3, 3), activation='relu',padding='same')(model)
#     # model = MaxPooling2D(pool_size=(2, 2))(model)
    
#     # model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
#     # model = MaxPooling2D(pool_size=(2, 2))(model)
    
#     # model = Flatten()(model) # 
#     # add fully-connected & dropout layers
#     x = Dense(2048, activation='relu',name='fc-1')(model)
#     # x = Dense(64, activation='relu',name='fc-1')(model)

#     x = Dense(64, activation='relu',name='fc-1')(model)
#     x = Dropout(0.5)(x)
    
#     # a softmax layer for 4 classes
#     out = Dense(nbClasses, activation='sigmoid',name='output_layer')(x)

#     # this is the model we will train
#     custom_resnet_model2 = Model(inputs=modelResNet.input, outputs=out)

#     custom_resnet_model2.summary()

#     # for layer in custom_resnet_model2.layers[:-6]:
#     #   layer.trainable = False

#     # custom_resnet_model2.layers[-1].trainable

#     return custom_resnet_model2

#     # imageShape =  Input(shape=IMAGE_INPUT_SHAPE)
#     # print("\n\n\nIn createModel(): ", imageShape, "\n\n")
#     # model = ResNet50(include_top=True, weights='imagenet', input_tensor=imageShape)
#     # model.summary()

#     # lastLayer = model.get_layer('avg_pool').output
    
#     # pprint(lastLayer)
#     # x = Flatten(name='flatten')(lastLayer)
#     # out = Dense(nbClasses, activation='sigmoid', name='output_layer')(x)
#     # custom_resnet_model = Model(inputs=image_input,outputs= out)
#     # custom_resnet_model.summary()

#     # for layer in custom_resnet_model.layers[:-1]:
#     #   layer.trainable = False

#     # custom_resnet_model.layers[-1].trainable

#     # return custom_resnet_model
