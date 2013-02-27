#!/usr/bin/python
import numpy as np
import cv2
from time import clock
import sys


def imageSize(image):
    im = cv2.imread(image)
    h,w = im.shape[:2]
    return h,w

def showImage(name):
	im = cv2.imread(name)
    cv2.imshow(name, im)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()

def normalize():
    im = cv2.imread('coltrane.jpg')
    h,w = im.shape[:2]
    print h,w
   
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    intim = cv2.integral(gray)
    newim = (255.0*intim)/ intim.max()	
    
    cv2.imshow('original_image', im)
    cv2.imshow('new_image', newim)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()

def fill():
    image = 'coltrane.jpg'
    im = cv2.imread(image)
    h,w = imageSize(image)
    
    diff = (6,6,6)
    mask = np.zeros((h+2,w+2),'uint8')
    cv2.floodFill(im,mask,(0,10), (100,255,0),diff,diff)
    
    showImage('original',im)
    





if __name__ == '__main__':
	showImage('ocean.jpg')



