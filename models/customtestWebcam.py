# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2
import multiprocessing
# loading the model
json_file = open('fer_70.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer_70.hdf5")
print("Loaded model from disk")
# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


cap = cv2.VideoCapture(0)
while True:
    
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    if not ret:
        break
    full_size_image = frame
    # loading image
    # full_size_image = cv2.imread('test_images/Unknown2.jpg')
    if full_size_image is None:
        raise SystemExit("Image does not exist. Please check filename.")
    # print("Image Loaded")
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    CLAHE_2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = CLAHE_2.apply(gray)

    # Using local copy of cascade to ensure it doesn't change or "update" during development
    FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = FACE_CASCADE.detectMultiScale(gray, 1.5, 3) #Attempt to find faces in the image
    largest_region = (0, 0, 0, 0)
    # print("Detected {} face(s). Using largest region.".format(len(faces)))
    for (x, y, w, h) in faces:
        if w > largest_region[2] and h > largest_region[3]:
            largest_region = x, y, w, h
    x, y, w, h = largest_region
    gray_roi = gray[y:y+h, x:x+w]

    if len(faces):
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_CUBIC), -1), 0)
        final_image = loaded_model.predict_proba(cropped_img)

        offset = 35
        x,y = 50,50

        # Print Emotions in order of Confidence
        # emotion_confidence = final_image.argsort().argsort()
        # print(emotion_confidence)
        # emotion_confidence = emotion_confidence.ravel()
        # print(emotion_confidence)
        # print(emotion_confidence)
        # print("Emotion: "+labels[int(np.argmax(final_image))])
        
        for index in range(7):
            # print(labels[index] + ': {0:.4f}%'.format(final_image[0][index] * 100))
            color = (0,0,0)
            if index == int(np.argmax(final_image)): color = (255,255,255)
            cv2.putText(frame, labels[index] + ": {0:.4f}%".format(final_image[0][index] * 100), (x, y+offset*index), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('capture', frame)
        cv2.waitKey(1)

        # Save image to folder
        emotion_max_label = labels[int(np.argmax(final_image))]
        savepath = "correct_images/" + emotion_max_label + "_correct.jpg"
        cv2.imwrite(savepath, gray_roi)
        # savepath = "wrong_images/" + emotion_max_label +"_wrong.jpg"
        # cv2.imwrite(savepath, full_size_image)
