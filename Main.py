#!/usr/bin/env python
# coding: utf-8

# # Training

# In[1]:


import time
from os import listdir
import numpy as np
import cv2 as cv


# In[2]:


sample_path = 'Samples/'
sample_names = listdir(sample_path)


# In[3]:


training_data, labels = [], []

for (i, sample_name) in enumerate(sample_names):
    image_path = sample_path + sample_name
    gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    training_data.append(np.array(gray, dtype = np.uint8))
    labels.append(i)
    
training_data = np.array(training_data)
labels = np.array(labels, dtype = np.int32)


# In[4]:


labels[2]


# In[5]:


training_data[0]


# In[6]:


model = cv.face.LBPHFaceRecognizer_create()


# In[7]:


model.train(training_data, labels)


# # Detection

# In[8]:


face_classifier_path = "haarcascade_frontalface_default.xml"
face_classifier = cv.CascadeClassifier(face_classifier_path)


# In[9]:


def face_detector(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    else:
        for face in faces:
            (x, y, w, h) = face
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 50, 50), 2)
            roi = image[y : y + h, x : x + w]
            resized = cv.resize(roi, (128, 128))
        return resized


# In[10]:


device = 0

capture = cv.VideoCapture(device)

while capture.isOpened():
    ret, frame = capture.read()
    face = face_detector(frame)
    
    try:
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        result = model.predict(gray)
        confidence = int(result[1])
        cv.putText(frame, str(confidence) + '%', (250,250), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 50), 2)
        
    except:
        cv.putText(frame, 'face not found', (250,250), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 50), 2)
        pass
    
    cv.imshow('result',frame)
    
    if cv.waitKey(1) == 13:
        break
        
capture.release()
cv.destroyAllWindows()


# In[11]:


capture.release()


# In[ ]:




