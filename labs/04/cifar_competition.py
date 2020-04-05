#!/usr/bin/env python
import argparse
import datetime
import os
import re

import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
import os, os.path
import pandas as pd
import math
import resource
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler

from cifar10 import CIFAR10

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.0003
        return lrate

    cifar = CIFAR10()

    (x_train, y_train), (x_test, y_test) = (cifar.train.data['images'], cifar.train.data['labels']) , (cifar.dev.data['images'], cifar.dev.data['labels'])

    #z-score
    num_classes = 10
    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)

    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        )
    datagen.fit(x_train)

    #training
    batch_size = 64

    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
    #                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
    #                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])

    #save to disk
    #model_json = model.to_json()
    #with open('model.json', 'w') as json_file:
    #    json_file.write(model_json)
    #model.save_weights('model.h5')

    model.load_weights('model.h5')

    #testing
    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open("cifar_competition_test.txt", "w", encoding="utf-8") as out_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
