import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, \
    BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.activations import relu
from keras.optimizers import Nadam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)

num_features = 64
num_labels = 7
batch_size = 256
epochs = 100 #Previous #50
width, height = 48, 48

x = np.load("fdataX.npy")
y = np.load("flabels.npy")

# x -= np.mean(x, axis=0)
# x /= np.std(x, axis=0)
# splitting into training, validation and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)


np.save('modXtest', X_test)
np.save('modytest', y_test)

# desinging the CNN
model = Sequential()

model.add(Conv2D(2*num_features, kernel_size=3, activation='linear', input_shape=(width, height, 1),
                 data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(AveragePooling2D(pool_size=3, strides=(2, 2)))


model.add(Conv2D(2*2*num_features, kernel_size=3, activation='linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(AveragePooling2D(pool_size=3, strides=(2, 2)))


model.add(Conv2D(2*2*2*num_features, kernel_size=3, activation='linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(AveragePooling2D(pool_size=3, strides=(2, 2)))


# model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Conv2D(2*2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))

model.add(Flatten())

# model.add(Dense(2*2*2*2*num_features, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2*2*2*num_features, activation='relu')))
# model.add(Dropout(0.4))
model.add(Dense(2*2*2*num_features, activation='linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))

model.add(Dense(2*2*num_features, activation='linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))


model.add(Dense(num_labels, activation='softmax'))


# model.summary()

# Compliling the model with adam optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-20),
              metrics=['accuracy'])



# Checkpoint
filepath = "checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = [EarlyStopping(monitor='val_loss',
                            patience=7,
                            verbose=1,
                            mode='auto',
                            restore_best_weights=True),
              ModelCheckpoint(filepath,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only=True,
                              mode='max', period=10)]
callbacks_list = checkpoint

# aug = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=False,
#     zoom_range=0.1)
#
# model.fit_generator(aug.flow(np.array(X_train), np.array(y_train), batch_size=batch_size),
#                     validation_data=(np.array(X_valid), np.array(y_valid)),
#                     steps_per_epoch=len(np.array(X_train)) // batch_size, epochs=epochs, callbacks=callbacks_list)

# training the model
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_list,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(y_valid)),
          shuffle=True)

# saving the  model to be used later
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print("Saved model to disk")
