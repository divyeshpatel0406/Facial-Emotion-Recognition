"""
Module for detecting faces within multiple images inside a folder and outputting the
results to an output folder.

Output images will have a blue rectangle around any faces detected.
The output will draw these rectangles on
the original, unaltered color image. Will also output the names of result images
where no face was detected or bad lighting is present.

Note the input folder and output folder paths must be altered to suit the environment.
"""

import glob
import time
import cv2

#PATHS FOR INPUT AND OUTPUT FOLDERS ARE REQUIRED
IMAGES = [cv2.imread(file) for file in glob.glob('faces/*jpg')] #Input path
PATH = 'faces/results/' #Output path

#Using local copy of cascade to ensure it doesn't change or "update" during development
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

#Contrast Limited Adaptive Histogram Equalization (CLAHE)
#is conditionally used as a pre-processing step
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))
CLAHE_2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

i = 0 #Keep track of which image we are working on (not actually used to iterate, only mark current)
j = 0 #Keep track of how many times we've run CLAHE, used to compute different time averages
k = 0 #Keep track of how many times we've run non-CLAHE
NORMAL_TIME = 0.0
CLAHE_TIME = 0.0
for img in IMAGES: #Go through each file in the folder
    start = time.time() #Start point for tracking runtime
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to gray

    #gray = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #gray = clahe.apply(gray)


    faces = FACE_CASCADE.detectMultiScale(gray, 1.08, 20) #Attempt to find faces in the image

    #If we have failed to detect a face, lighting conditions are probably sub-optimal
    #The reasoning for this re-run is due to the assumption that most images will have at least
    #decent lighting conditions. Therefore, running CLAHE to every case is usually
    # not necessary. The more times we don't use CLAHE, the more processing time we
    # save and the more responsive the program will feel
    if not list(faces):
        #Apply Contrast Limited Adaptice Histogram Equalization and detect faces
        gray2 = CLAHE_2.apply(gray)
         #Look for large rectangles only to avoid false positives
        faces2 = FACE_CASCADE.detectMultiScale(gray2, 1.03, 15, minSize=(40, 40))

        #If we fail to detect a face a second time, print the image number
        if not list(faces2):
            print("No face found " + str(i))
        else:
            print("Bad lighting " + str(i))
        #Draw a rectangle around the faces detected in the image
        for (x, y, w, h) in faces2:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        j += 1
        CLAHE_TIME += time.time() - start #Add runtime to running total

    else:
        #Draw a rectangle around the faces detected in the image
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #This is the Region of Interest for our image (The area of the face)
            roi_gray = gray[y:y+h, x:x+w]
        NORMAL_TIME += time.time() - start #Add runtime to running total
        k += 1

    #Write the altered image to the results folder
    cv2.imwrite(PATH + 'pic %d.jpg'%i, img)
    i += 1

#Hold average runtimes
NORMAL_TIME_AVG = 0.0
CLAHE_TIME_AVG = 0.0

#Make sure not to divide by zero when computing averages
if k != 0:
    NORMAL_TIME_AVG = NORMAL_TIME / k
if j != 0:
    CLAHE_TIME_AVG = CLAHE_TIME / j

#Print averages for different conditions
print("Normal lighting average runtime: " + str(NORMAL_TIME_AVG))
print("Bad lighting average runtime: " + str(CLAHE_TIME_AVG))
print("Total average runtime: " + str((NORMAL_TIME + CLAHE_TIME) / i))

#Garbage cleanup
cv2.destroyAllWindows()
