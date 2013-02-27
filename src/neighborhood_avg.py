#!/usr/bin/python
import numpy as np
import cv2
from time import clock
import sys


if __name__ == '__main__':
    im = cv2.imread('ocean.jpg')
    I = np.asarray(cv2.imread('ocean.jpg'))
   
    newI = np.zeros_like(I)    
    newI.resize((I.shape[0]+2,I.shape[1]+2,I.shape[2])) 
   
    #pad the zeros for each channel(k); TODO:pad based on size of neighborhood 
    for k in xrange(I.shape[2]):    
        for j in xrange(1,newI.shape[1]-1):
            for i in xrange(1,newI.shape[0]-1):
                newI[i,j,k] = I[i-1,j-1,k]
    print newI.shape,'\n'
     
    b= np.zeros_like(newI)
    
    for chan in xrange (2,I.shape[2]):
        for h in xrange(1, newI.shape[1]-1):
            for v in xrange(1,newI.shape[0]-1):         
                a = newI[v-1:v+2,h-1:h+2,chan]       
                
                b[v,h,chan] = int(a.mean())
    
           
    final_array = b[1:b.shape[0]-1,1:b.shape[1]-1,:] 
      
    #final_array = cv2.blur(I,(6,6))    
    cv2.imshow('original_image', im)
    cv2.imshow('final_image', final_array)    
    cv2.waitKey(8000)
    cv2.destroyAllWindows()    
  
 
  
    
    



    


    
