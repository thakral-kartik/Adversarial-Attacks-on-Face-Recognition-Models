# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:51:19 2019

@author: Kartik
"""

from glob import glob
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import openface
from keras_vggface.vggface import VGGFace
from openface.align_dlib import AlignDlib #align_dlib.py was explicitly copied in the site-packages of openface and we want to import a class "AligbDlib" from that .py file
import dlib
import random
#from inception_resnet_v1 import * #inception_resnet_v1.py was also copied in the current directory
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.models import load_model
from matplotlib import pyplot as plt


def get_rep(x):
    
    global facenet_model
    x_prime=[]
    for img in x:
        face_pixels = img.astype('float32') # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0) #adding a new dimension as InceptionNetV1 needs it in this way only (1,160,160,3)
        rep = facenet_model.predict(samples)
        x_prime.append(rep[0])
    
    return x_prime


def get_vggface_rep(x,y):
    model=VGGFace(model='vgg16')
    crop_dim=224
    k=-1
    x_prime,y_prime=[],[]
    for img in x:
        boundary_box = align_dlib.getLargestFaceBoundingBox(img) #building a bounding box around the face in the image
        k+=1
        if boundary_box is not None:
            aligned = align_dlib.align(crop_dim, img, boundary_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP) #cropping the image from eyes to chin
            face_pixels = aligned.astype('float32') # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            samples = np.expand_dims(face_pixels, axis=0) #adding a new dimension as InceptionNetV1 needs it in this way only (1,160,160,3)
            rep = model.predict(samples)
            x_prime.append(rep[0])
            y_prime.append(y[k])
            print(k,end=" ")
        else:
            print("found fault!",end=" ")
    return np.array(x_prime),np.array(y_prime)
