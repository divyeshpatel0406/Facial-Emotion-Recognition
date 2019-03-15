"""
This code is for basic Transfer Learning.

It uses a MobileNetV2 model pre-trained on the ImageNet dataset containing 1000 classes and transfers.

"""


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import mobilenetv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


img_width, img_height = 224, 224
train_data_dir = "train"
validation_data_dir = "val"
nb_train_samples = 3200
nb_validation_samples = 800
batch_size = 16
epochs = 20

model = mobilenetv2.MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers:      # model.layers[:x] for freezing first x layers
    layer.trainable = False

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])

# Initiate the train and test generators with data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   zoom_range=0.3,
                                   rotation_range=30)

test_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip = True,
                                  zoom_range = 0.3,
                                  rotation_range=30)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_height, img_width),
                                                        class_mode="categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("mobilenetv2.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model
model_final.fit_generator(train_generator,
                          samples_per_epoch=nb_train_samples,
                          epochs=epochs,
                          validation_data=validation_generator,
                          nb_val_samples=nb_validation_samples,
                          callbacks=[checkpoint, early])

model_final.save_weights('trained_weights_new.h5')
