import io
import numpy as np
import vtk
import whitematteranalysis as wma

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.layers import Conv1D, MaxPooling1D

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def CNN_simple(x_train, y_train, x_validation, y_validation, num_classes, out_path, data_augmentation=False):
    """The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""

    batch_size = 512
    epochs = 100
    data_augmentation = False

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')

    earlyStopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00, patience=10, verbose=0, mode='auto')
    net_save = ModelCheckpoint('./{}/best_weights.h5'.format(out_path), save_best_only=True, monitor='val_accuracy',
                               mode='auto')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_validation, y_validation),
                  callbacks=[earlyStopping, net_save],
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        print('TBD!')

    return model


def CNN_simple_1D(x_train, y_train, x_test, y_test, num_classes, data_augmentation=False):
    """The most simple feature for initial test

	Parameters
	----------
	TODO:

	"""
    max_features = 50
    maxlen = 45
    batch_size = 32
    embedding_dims = 5
    filters = 32
    kernel_size = 3
    hidden_dims = 20
    epochs = 10

    print('Loading data...')

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(32, 3,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(Conv1D(32,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    ## # We add a vanilla hidden layer:
    ## model.add(Dense(128))
    ## model.add(Dropout(0.2))
    ## model.add(Activation('relu'))

    ## model.add(Dense(num_classes, activation='softmax'))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # print(model.summary())

    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

    return model


def predict(model, x_data, y_data=None, y_name=None, verbose=False):
    y_prediction = model.predict_classes(x_data)

    if y_data is not None:

        try:
            prediction_report = classification_report(y_data, y_prediction, target_names=y_name)

            if verbose:
                print(prediction_report)

            con_matrix = confusion_matrix(y_data, y_prediction)
            if verbose:
                print(con_matrix)

        except ValueError:
            print('[Warning]: There is missing tract')
            print(np.unique(y_prediction))
            prediction_report = None
            con_matrix = None

    else:
        prediction_report = None
        con_matrix = None

    return y_prediction, prediction_report, con_matrix
