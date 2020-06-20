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




def get_dataset():
    #creating a dictionary of images where dict is of form : 
    #{'Person_name': matrices of all images in that folder}
    images = {}
    cwd = os.getcwd() + "\\modified_dataset_158\\"
    for folder in glob(cwd+"\\*"):
        key = (folder.split('\\'))[-1]
        l = []
        for image in glob(folder+"\\*"):
            img = cv2.imread(image)
            l.append(img)
        images[key] = l
    return images

def shuffle_and_split(x, y):
    #shuffle
    x, y = shuffle(x, y)
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.3, random_state = 0)
    return x_train, x_test, y_train, y_test

def split_dataset(data):
    count = 0
    label_mapping = {}
    x, y = [], []
    #x_train, y_train, x_test, y_test = [], [], [], []
    for k,v in data.items():
        '''
        #training: 100 folders, test: 58 folders
        if count >= 100:
            #test split
            x_test+=v    
            for _ in range(len(v)):
                y_test.append(count)
        else:
            #training split
            x_train+=v  #extending all the images corresponding to a person into a single list    
            for _ in range(len(v)):
                y_train.append(count)  #maintaining label corresponding to each image of the folder
        '''
        label_mapping[count] = k    #maintaining a mapping {number: 'Person_name'}
        count+=1
        #Taking max 20 images from each folder to deak with imbalance dataset
        l=[]
        lim = 52    #initiallly lim = 20 for folder size of 10 in mod_dataset.py
        '''if len(v)>lim:
            x += v[:lim]
            l = [count]*lim
        else:
            x+=v
            l=[count]*len(v)
        y.extend(l)'''
        
        #for modified script with 10 classes and considering all images from each folder
        x += v
        l = [count]*len(v)
        y.extend(l)
        
    #return shuffle_and_split(x, y)
    return x, y, label_mapping
    #return x_train, y_train, x_test, y_test
