# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import threading
import numpy
import os, sys
import numpy as np
import cv2
import multiprocessing
from queue import Queue
# loading the model
model_filename = 'fer_70.json'
weights_filename = 'fer_70.hdf5'
cascade_filename = 'haarcascade_frontalface_default.xml'

import sys, os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the pyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_filename = os.path.join(application_path, model_filename)
weights_filename = os.path.join(application_path, weights_filename)
cascade_filename = os.path.join(application_path, cascade_filename)
json_file = open(model_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_filename)
print("Loaded model from disk")
# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

vs = WebcamVideoStream()
vs.start()
while True:
    frame = vs.read()
    # cv2.imshow("capture", frame)

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
    FACE_CASCADE = cv2.CascadeClassifier(cascade_filename)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.6, 3) #Attempt to find faces in the image
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
        

        # Save image to folder
        emotion_max_label = labels[int(np.argmax(final_image))]
        savepath = "correct_images/" + emotion_max_label + "_correct.jpg"
        cv2.imwrite(savepath, gray_roi)
        # savepath = "wrong_images/" + emotion_max_label +"_wrong.jpg"
        # cv2.imwrite(savepath, full_size_image)
    if(cv2.waitKey(1) % 256 == 27):
        vs.stop()
        cv2.destroyAllWindows()
        raise SystemExit("Escape key pressed. Exiting program.")
