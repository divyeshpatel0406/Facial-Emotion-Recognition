# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2

#loading the model
json_file = open('fer_70.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer_70.hdf5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#loading image
full_size_image = cv2.imread("test_images/Happy1.jpg")
print("Image Loaded")
gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)

##Haar Cascade Code  goes here.
cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
final_image = loaded_model.predict_proba(cropped_img)


#Print Emotions in order of Confidence
# emotion_confidence = final_image.argsort().argsort()
# print(emotion_confidence)
# emotion_confidence = emotion_confidence.ravel()
# print(emotion_confidence)
#print(emotion_confidence)
print("Emotion: "+labels[int(np.argmax(final_image))])
for index in range(6):
	print(labels[index] + ': {0:.4f}%'.format(final_image[0][index] * 100))
cv2.imshow('Emotion', full_size_image)
cv2.waitKey()

#Save image to folder
emotion_max_label = labels[int(np.argmax(final_image))]
savepath = "correct_images/" + emotion_max_label + "_correct.jpg"
cv2.imwrite(savepath, full_size_image)
#savepath = "wrong_images/" + emotion_max_label +"_wrong.jpg"
#cv2.imwrite(savepath, full_size_image)
