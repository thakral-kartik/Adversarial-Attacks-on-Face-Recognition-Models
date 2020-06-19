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


def generate_gaussian_noise(row, col, channel):
    mean = 0.0
    variance = 0.3 #1.5
    sigma = variance**0.5
    gauss = np.array((row, col))
    gauss = np.random.normal(mean,sigma,(row,col,channel))
    gauss = gauss.reshape(row,col,channel)
    #noisy = image + gauss
    return gauss.astype('uint8')

def add_noise(x, noise):   #adding Gaussian noise to each image matrix
    #noise = np.multiply(0.007, noise)
    #return x
    return np.add(x,noise)

def generate_random_noise(x_test):
    #row, col, channel = x_test.shape[0], x_test.shape[1], x_test.shape[2]
    #noise = generate_gaussian_noise(x_test[0].shape[0], x_test[0].shape[1], x_test[0].shape[2])
    #noise = np.reshape(np.zeros(x_test[0].shape), (x_test[0].shape[0], x_test[0].shape[1], x_test[0].shape[2]))
    noise = np.reshape(np.zeros(160*160*3), (160, 160, 3))
    
    coord = []
    no_of_coord = 80
    for i in range(3):
        for j in range(no_of_coord):
            c = random.randint(0,noise.shape[0]-1), random.randint(0,noise.shape[1]-1)
            coord.append(c)
    
    for z in range(3):
        for i in range(no_of_coord):
            noise[coord[i][0]][coord[i][1]][z] = 100 
    return noise

def generate_grid_noise(x_test):
    noisy = []#np.reshape(np.zeros(160*160*3), (160, 160, 3))
    coord = []
    no_of_coord = random.randint(1,2)
    for i in range(no_of_coord):
        l=[]
        l.append((random.randint(5, 155), 0)) #(x1, 0)
        l.append((random.randint(5, 155), 155)) #(x1', H)
        l.append((0, (random.randint(5, 155)))) #(0, y1)
        l.append((155, (random.randint(5, 155)))) #(W, y1')
        coord.append(l)
    for im in x_test:
        img = im.copy()
        for z in range(no_of_coord):
            cv2.line(img, coord[z][0], coord[z][1], (0,0,0), 1)
            cv2.line(img, coord[z][2], coord[z][3], (0,0,0), 1)
        noisy.append(img)
    return noisy
                    
def generate_eye_patch(x_test):
    noisy = []
    
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    #print(eye_cascade)
    
    for image in x_test:
        img = image.copy()
        #img = cv2.imread('image_new.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),-1)
        noisy.append(img)
    
    cv2.imshow('img', noisy[-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return noisy


def add_spe(row,col,ch,image):
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy
