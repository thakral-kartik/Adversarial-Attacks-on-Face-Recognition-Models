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


def get_align(x,y):
    x_test,y_test=[],[]
    global alogn_dlib,facenet_model
    
    crop_dim=160
    k=-1
    for img in x :
        boundary_box = align_dlib.getLargestFaceBoundingBox(img) #building a bounding box around the face in the image
        k+=1
        if boundary_box is not None:
            aligned = align_dlib.align(crop_dim, img, boundary_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP) #cropping the image from eyes to chin
            x_test.append(aligned)
            #if k !=1287:
            y_test.append(y[k])
            print(k,end=" ")
        else:
            print("found fault!",end=" ")
    
    return x_test,y_test
        

def align_dataset(x_train,y_train,x_test,y_test):
    x_prime,y_prime,x_noisy=[],[],[]
    
    global align_dlib,facenet_model
    
    k=-1
    x=list(x_train)+list(x_test)
    y=list(y_train)+list(y_test)
    loc=len(list(y_train))
    crop_dim=160
    
    for img in x :
        boundary_box = align_dlib.getLargestFaceBoundingBox(img) #building a bounding box around the face in the image
        k+=1
        if boundary_box is not None:
            aligned = align_dlib.align(crop_dim, img, boundary_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP) #cropping the image from eyes to chin
            #face_pixels = aligned.astype('float32') # standardize pixel values across channels (global)
            x_prime.append(aligned)
            y_prime.append(y[k])
            print(k,end=" ")
        else:
            print("found fault!",end=" ")
    x_train, y_train = x_prime[:loc-1], y_prime[:loc-1]
    x_test, y_test = x_prime[loc-1:], y_prime[loc-1:]
    #plt.imshow(x_noisy[0])
    return x_train, y_train, x_test, y_test
