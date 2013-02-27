#!/usr/bin/python

import numpy as np
import cv2

from time import clock
import sys



if __name__ == '__main__':
    
      
        

    im = cv2.imread('coltrane.jpg')
      
    mul_im = cv2.multiply(im,np.array([float(3)]))    

    #add 2**(user input) values to every pixel of the array
    new_im = cv2.add(mul_im, np.array([2**(int(sys.argv[1]))]))
    

    cv2.imshow('original_image', im)
    cv2.imshow('new_image', new_im)
    
    cv2.waitKey(4000)
    cv2.destroyAllWindows()
  
