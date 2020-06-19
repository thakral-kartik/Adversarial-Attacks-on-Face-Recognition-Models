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

from preprocessing import *
from get_rep import *
from dataloader import *
from noise import *


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))
    
def align_crop_bb_embedding_normalize(x_train, y_train, x_test, y_test):
    x_prime, y_prime, x_noisy = [], [], []
    crop_dim = 160 #bcoz model needs it in (1,160,160,3) shape in RGB format and output is (1,128) embedding vectors
    #model = InceptionResNetV1()
    
    global align_dlib, facenet_model
    
    k=-1
    x = list(x_train) + list(x_test)
    y = list(y_train) + list(y_test)
    loc = len(list(y_train))

    dimpy=[]
    norm_images=[]
    for img in x:
        #img=cv2.imread(image)
        boundary_box = align_dlib.getLargestFaceBoundingBox(img) #building a bounding box around the face in the image
        k+=1
        if boundary_box is not None:
            aligned = align_dlib.align(crop_dim, img, boundary_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP) #cropping the image from eyes to chin
            #new_img = np.expand_dims(aligned, axis=0)   
            #rep = l2_normalize(facenet_model.predict(new_img)) #model.predict will give the (1,128) embedding vectors and they are then normalized
            #noise = generate_gaussian_noise(aligned.shape[0],aligned.shape[1], aligned.shape[2])
            #noise = generate_random_noise(x)
            
            #print(aligned.shape)
            #x_noisy.append(add_noise(aligned, noise))
            x_noise = generate_grid_noise(x)
            
            face_pixels = aligned.astype('float32') # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            
            
            norm_images.append(add_noise(face_pixels, noise))
            samples = np.expand_dims(face_pixels, axis=0) #adding a new dimension as InceptionNetV1 needs it in this way only (1,160,160,3)
            
            dimpy.append(samples)
            rep = facenet_model.predict(samples)
            #print(rep[0].shape)
            x_prime.append(rep[0])
            y_prime.append(y[k])
            print(k,end=" ")
        else:
            print("found fault!",end=" ")
    x_train, y_train = x_prime[:loc-1], y_prime[:loc-1]
    x_test, y_test = x_prime[loc-1:], y_prime[loc-1:]
    dimpy_train,dimpy_test=dimpy[:loc-1],dimpy[loc-1:]
    #plt.imshow(x_noisy[0])
    x_noisy_train = x_noisy[:loc-1]
    #plt.imshow(x_noisy_train[0])
    x_noisy_test = x_noisy[loc-1:]
    norm_train_noise=norm_images[:loc-1]
    norm_test_noise=norm_images[loc-1:]
    return x_train, y_train, x_test, y_test, x_noisy_train, x_noisy_test,dimpy_train,dimpy_test,norm_train_noise,norm_test_noise
    

def aligned(x,y):
    x_prime=[]
    y_prime=[]
    crop_dim = 160 #bcoz model needs it in (1,160,160,3) shape in RGB format and output is (1,128) embedding vectors
    global align_dlib, facenet_model
    k=-1
    for img in x:
        #img=cv2.imread(image)
        #boundary_box = align_dlib.getLargestFaceBoundingBox(img) #building a bounding box around the face in the image
        k+=1
        #if boundary_box is None:
        #    print("found fault!",end=" ")
        #else:    
        #aligned = align_dlib.align(crop_dim, img, boundary_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP) #cropping the image from eyes to chin
        #face_pixels = aligned.astype('float32') # standardize pixel values across channels (global)
        face_pixels = img.astype('float32') # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
          #the above lines are added by Dimpy because we are getting the unnormalised images.  
        samples = np.expand_dims(face_pixels, axis=0)
        rep = facenet_model.predict(samples)
        x_prime.append(rep[0])
        y_prime.append(y[k])
        
    return x_prime,y_prime




def aligned_test(x,y):
    x_prime=[]
    y_prime=[]
    crop_dim = 224#bcoz model needs it in (1,160,160,3) shape in RGB format and output is (1,128) embedding vectors
    global align_dlib, facenet_model
    k=-1
    for img in x:
        #img=cv2.imread(image)
        boundary_box = align_dlib.getLargestFaceBoundingBox(img) #building a bounding box around the face in the image
        k+=1
        if boundary_box is None:
            print("found fault!",end=" ")
        else:    
            aligned = align_dlib.align(crop_dim, img, boundary_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP) #cropping the image from eyes to chin
            face_pixels = aligned.astype('float32') # standardize pixel values across channels (global)
            face_pixels = img.astype('float32')
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            x_prime.append(face_pixels)
            y_prime.append(y[k])
        
    return x_prime,y_prime

def SVM_clf(x_train, y_train, x_test, y_test):
    #x_train, x_test, y_train, y_test = shuffle_and_split(x, y)
    in_encoder = Normalizer(norm='l2')
    train_x = in_encoder.transform(x_train)
    test_x = in_encoder.transform(x_test)
    #out_encoder = LabelEncoder()
    #out_encoder.fit(y_train)
    #train_y = out_encoder.transform(y_train)
    #test_y = out_encoder.transform(y_test)
    train_y,test_y=np.array(y_train),np.array(y_test)
    clf = SVC(kernel='linear',probability=True)
    clf.fit(train_x, train_y)
    print("SVM accuracy: ",clf.score(test_x, test_y))
    return clf, train_x, train_y, test_x, test_y    




def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    #euclidean_distance = l2_normalize(euclidean_distance )
    return euclidean_distance


#main
'''
images = get_dataset()
#x_train, x_test, y_train, y_test = split_dataset(images)
x, y, label_mapping = split_dataset(images)
X_train, X_test, Y_train, Y_test = shuffle_and_split(x, y)

np.save('X_train_158.npy',X_train)
np.save('Y_train_158.npy',Y_train)
np.save('X_test_158.npy',X_test)
np.save('Y_test_158.npy',Y_test)

'''
X_test=np.load('X_test_158.npy')
Y_test=np.load('Y_test_158.npy')
X_train=np.load('X_train_158.npy')
Y_train=np.load('Y_train_158.npy')


align_dlib = AlignDlib('shape_predictor_68_face_landmarks.dat') #calling constructor for aligning the image
facenet_model = load_model('facenet_keras.h5') #facenet weights from keras lib

x_train, y_train = aligned_test(X_train, Y_train)
x_test, y_test = aligned_test(X_test, Y_test)

x_noisy_test = generate_grid_noise(x_test)


    
'''
x_train, y_train, x_test, y_test, x_noisy_train, x_noisy_test, dimpy_tran, dimpy_test, norm_train_noisy, norm_test_noisy = align_crop_bb_embedding_normalize(X_train, Y_train, X_test, Y_test)



from keras.models import Model
for layer in facenet_model.layers:
    print(layer.name)
layer_name='Conv2d_2b_3x3'
layer_name='Block8_6_Branch_1_Conv2d_0c_3x1_Activation'
layer_name='Conv2d_4b_3x3'
layer_name='Block8_6_Branch_1_Conv2d_0a_1x1'
intermediate_layer_model = Model(inputs=facenet_model.input,
                                 outputs=facenet_model.get_layer(layer_name).output)


#print(len(x),len(y))
print("svm started")

from sklearn.externals import joblib

clf, x_train, y_train, x_test, y_test = SVM_clf(x_train, y_train, x_test, y_test)
joblib.dump(clf, 'svm_model_10.sav')
X_TEST,Y_TEST=aligned(x_noisy_test,y_test)
print("Classification score after adding noise: ", clf.score(X_TEST, Y_TEST))

'''

# for VGGFACE uncomment below lines
'''
x_train,y_train,x_test,y_test=align_dataset(X_train,Y_train,X_test,Y_test)
X_Train=get_vggface_rep(x_train)
X_Test=get_vggface_rep(x_test)
'''
align_dlib=AlignDlib('shape_predictor_68_face_landmarks.dat')

x_train,y_train,x_test,y_test=align_dataset(X_train,Y_train,X_test,Y_test)
X_Train=get_rep(x_train)
X_Test=get_rep(x_test)
clf, x_train_, y_train_, x_test_, y_test_ = SVM_clf(X_train, y_train, X_test, y_test)

noisy_eye_patch=generate_eye_patch(X_test)
noisy_eye_patch_a,y_test_e_p=get_align(noisy_eye_patch,Y_test)
noisy_x_test_eye_rep=get_rep(noisy_eye_patch_a)

print("Score on eye patch noise :",clf.score(noisy_x_test_eye_rep,y_test_e_p))
noisy_grid_patch=generate_grid_noise(x_test)
noisy_x_test_grid_rep=get_rep(noisy_grid_patch)

print("Score on eye patch noise :",clf.score(noisy_x_test_grid_rep,y_test))
