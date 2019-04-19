#!/usr/bin/env python
# coding: utf-8

# In[8]:


from scipy import ndimage, misc
import numpy as np
import os
import cv2
import warnings
import base64
import csv
import json
import base64
from PIL import Image
# 0: -4593 images- Angry
# 1: -547 images- Disgust
# 2: -5121 images- Fear
# 3: -8989 images- Happy
# 4: -6077 images- Sad
# 5: -4002 images- Surprise

def main():
    outPath = r'C:\CSProject\fer2013.csv'
    path = r'C:\CSProject\train\sad'

    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)
        img = Image.open(input_path).convert('L')  # convert image to 8-bit grayscale
        img = img.resize((48,48))
        #newline=''
        WIDTH, HEIGHT = img.size
        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        # At this point the image's pixels are all in memory and can be accessed
        # individually using data[row][col].
        imagePixels = " "
        # For example:
        for row in data:
            imagePixels +=' '.join('{:3}'.format(value) for value in row)
        
        with open(r'C:\CSProject\fer2013.csv', 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(('4',imagePixels,'Training'))
    print('It\'s saved!')


if __name__ == '__main__':
    main()
    


# In[ ]:




