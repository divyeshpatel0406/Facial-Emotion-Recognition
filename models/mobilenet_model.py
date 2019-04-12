import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from PIL import Image 
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

base_model=MobileNet(weights='imagenet',include_top=False) 

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)  #1024
x=Dense(512,activation='relu')(x)   #512
preds=Dense(5,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

#./train/ should be a folder containing 5 folders named after each emotion, with each folder filled with pictures of said emotion
train_generator=train_datagen.flow_from_directory('./train/',
                                                 target_size=(48,48),
                                                 color_mode='rgb',
                                                 batch_size=20,
                                                 class_mode='categorical',
                                                 shuffle=True)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


#filepath = "checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath,
#			monitor='val_acc',
#			verbose=1,
#			save_best_only=True,
#			mode='max')
#callbacks_list = [checkpoint]

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   	steps_per_epoch=step_size_train,
					epochs=5)
					#callbacks=callbacks_list,
					#verbose=1)

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print("Saved model to disk")
