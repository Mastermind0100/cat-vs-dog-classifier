"""
Created on Mon Jun  3 11:44:34 2019

@author: AtharvaHudlikar
"""
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

trdata = 2000
vltdata = 800
batch = 16

training_data = 'insert training data directory here'
validation_data = 'insert validation data directory here'

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen=ImageDataGenerator(rescale = 1./255)

training_set=train_datagen.flow_from_directory(directory = training_data,
                                                 target_size = (64, 64),
                                                 batch_size = batch,
                                                 class_mode = 'binary')

test_set=test_datagen.flow_from_directory(directory = validation_data,
                                            target_size = (64, 64),
                                            batch_size = batch,
                                            class_mode = 'binary')

model.fit_generator(training_set,steps_per_epoch = 125,         
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 50)                 #trdata//batch = 125
                                                                #vldata//batch = 50

model.save('modelwts.h5')
