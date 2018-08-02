import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import random, sys, keras

def deepsensing_network(in_shp = [2, 128], classes = ['busy' ,'idle']):
    K.set_image_dim_ordering('th')
    dr = 0.5
    model = models.Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense( len(classes), init='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #model.summary()
    return model

def deepsensing_train(datafile, EbN0, in_shp=[2,128], classes=['busy', 'idle'], nb_epoch=100, batch_size=1000):
    from util import dataset_load
    [X_train, Y_train, X_test, Y_test] = dataset_load(datafile)
    
    model = deepsensing_network(in_shp, classes)
    model_saved_path = 'QPSK.wts_' + str(EbN0) + '.h5'
    
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=2,
                        validation_data=(X_test, Y_test),
        callbacks = [
            keras.callbacks.ModelCheckpoint(model_saved_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ])

    model.load_weights(model_saved_path)
    
    X = [X_train, Y_train, X_test, Y_test]
    
    return model, model_saved_path, X

def deepsensing_load_model(modelfile, in_shp=[2,128], classes=['busy', 'idle']):
    model = deepsensing_network(in_shp, classes)
    model.load_weights(modelfile)
    return model