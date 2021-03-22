#!/usr/bin/env python
# coding: utf-8

# In[5]:


import time
import numpy as np
import cv2 as cv


# In[6]:


face_classifier_path = "haarcascade_frontalface_default.xml"
face_classifier = cv.CascadeClassifier(face_classifier_path)


# In[7]:


def face_extractor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    else:
        for face in faces:
            (x, y, w, h) = face
            croped_image = image[y : y + h, x : x + w]
            
        return croped_image


# In[13]:


sample_path = "Samples/"
n_sample = 500
count = 0
device = 0
capture = cv.VideoCapture(device)

while capture.isOpened():
    ret, frame = capture.read()
    croped_image = face_extractor(frame)
    
    if croped_image is not None:
        resized = cv.resize(croped_image, (128, 128))
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        cv.imwrite(sample_path + str(count) + '.jpg', gray)
        cv.putText(croped_image, str(count), (25,25), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 50), 2)
        count += 1
    else:
        print("Face Not Found")
        croped_image = np.zeros(shape = (256, 256, 3), dtype = np.int16)
        
    cv.imshow('Result', croped_image)
    
    if cv.waitKey(1) == 13 or count == n_sample:
        break
        
capture.release()
cv.destroyAllWindows()


# In[ ]:




