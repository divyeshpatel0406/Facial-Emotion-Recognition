#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import warnings
import base64
import csv
import json
import base64

#image = pd.read_csv
#create image base


#create array columns to be inserted into csv
#in this example i am inputting a sad image 


# 0: -4593 images- Angry
# 1: -547 images- Disgust
# 2: -5121 images- Fear
# 3: -8989 images- Happy
# 4: -6077 images- Sad
# 5: -4002 images- Surprise

# from PIL import Image
# im = Image.open(r'C:\CSProject\sad_Selfie_gs.jpg')
# #pixels = list(im.getdata())
# pixels = list(im.getdata())
# #create item to write to csv


from PIL import Image

img = Image.open(r'C:\CSProject\sad_Selfie_gs.jpg').convert('L')  # convert image to 8-bit grayscale
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

# # Here's another more compact representation.
# chars = '@%#*+=-:. '  # Change as desired.
# scale = (len(chars)-1)/255.
# print()
# for row in data:
#     print(' '.join(chars[int(value*scale)] for value in row))
    
    
    
with open(r'C:\CSProject\fer2013.csv', 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(('4',imagePixels,'Training'))
    print('It\'s saved!');
    

#change image string to an image.show()
#image = '1 2 3 4 5 6'
#image_width, image_height = 2, 3
#result = np.fromstring(image, dtype=int, sep=" ").reshape((image_height, image_width))
#results.show



# In[ ]:




