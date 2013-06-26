#!/usr/bin/python
import numpy as np
import cv2
from time import clock
import sys
import math
#setting channel to 0,1,2
channel = 0
#original image setting
image = 'lemonde.jpg'


def NeighborhoodAvging():
    #read the image as a numpy array
    I = np.asarray(cv2.imread(image))
    #setup an all zero array with the same size of I
    newI = np.zeros_like(I)
    #zero-pad the side of the array depending on the size of the chosen neighborhood (2 here for a 3x3)    
    newI.resize((I.shape[0]+2,I.shape[1]+2,I.shape[2])) 
    #pad the zeros for each channel(k); TODO:pad based on size of neighborhood 
    #fill up the array with data from the image array
    for k in xrange(I.shape[2]):    
        for j in xrange(1,newI.shape[1]-1):
            for i in xrange(1,newI.shape[0]-1):
                newI[i,j,k] = I[i-1,j-1,k]
    print newI.shape,'\n' 
    b= np.zeros_like(newI)
    #pick the channel
    for chan in xrange (channel,I.shape[2]):
        #iterate horizontally
        for h in xrange(1, newI.shape[1]-1):
            #iterate vertically
            for v in xrange(1,newI.shape[0]-1):
                #extract the pixel and its neighborhood         
                a = newI[v-1:v+2,h-1:h+2,chan]       
                #take the mean and copy to a temp array b
                b[v,h,chan] = int(a.mean())
    
    #chop the unwanted fring pixels added due to zero padding earlier      
    final_array = b[1:b.shape[0]-1,1:b.shape[1]-1,:]
    return final_array 


def guassianBlur(image):
    return cv2.GaussianBlur(np.asarray(cv2.imread(image)), (51,51), 1)
    
def binThresholdAndGuassianBlur(image):
    return cv2.threshold(cv2.GaussianBlur(np.asarray(cv2.imread(image)), (51,51), 1), 230, 255, 0)

#show original and final results
def showImage(original, final):
    cv2.imshow('original_image', original)
    cv2.imshow('final_image', final)    
    cv2.waitKey(2000000)
#    cv2.destroyAllWindows() 


class AdaptiveThreshold():
    def __init__(self, im = 0,globalMean = 0, globalSD = 0, windowSize = 9, localMean = 0, localSD = 0, k = 0, threshold = 0):
        if im:
            self.im = im
        else:
            #read the image as grayscale and find its matrix representation 
            self.im = np.asarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        
        if globalMean:
            self.globalMean = globalMean
        else:
            #calculate the global mean
            self.globalMean = im.mean()
        
        if globalSD:
            self.globalSD = globalSD
        else:
            #calculate the global standard deviation
            self.globalSD = self.im.std()

#adaptive thresholding based on NiBlack method
#@TODO: Class implementation
def adaptiveThreshold(image):
    #im = cv2.imread(image)
    windowSize = 75
    #convert image to gray scale
    grayImage = image
    #calculate global mean
    print grayImage.mean()
    #calculate global standard deviation
    print grayImage.std()
    #calculate the local means into meanArray
    meanArray = cv2.blur(grayImage, (windowSize,windowSize))
    #calculate the local standard deviations into sdvArray
    sdvArray = np.sqrt(cv2.pow(np.subtract(meanArray,grayImage.mean()),2)/windowSize)
    #calculate the local "ks" into Karray
    kArray = -0.3 * ((np.subtract(grayImage.mean() * grayImage.std(), np.multiply(meanArray, sdvArray))) / np.maximum(grayImage.mean() * grayImage.std(), np.multiply(meanArray, sdvArray)))
    #finally find the local thresholds into thresholdArray
    thresholdArray = np.add(meanArray, np.multiply(kArray, sdvArray))
    #iterate over the image's array to get each pixel
    pixel = np.nditer(grayImage, flags=['multi_index'])
    #set the thresholdImage
    thresholdImage = grayImage
    #iterate over thresholds
    for threshold in np.nditer(thresholdArray):
        #binarization based on local threshold
        if pixel.value <= threshold:
            thresholdImage[pixel.multi_index] = 255
        else:
            thresholdImage[pixel.multi_index] = 0
        #iterate to the next pixel
        pixel.next()
         
#    grayImage = np.asarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    #finalImage = np.subtract(grayImage, thresholdImage) 
    #showImage(grayImage, thresholdImage)
    return thresholdImage
    
def otsuThreshold(image, threshold):
    #read image in gray scale
#    im = cv2.imread(image,0)
    #binarization using Otsu's algorithm
    thresholdImage = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholdImage = thresholdImage[1]
#    showImage(im, thresholdImage)
    return thresholdImage
 
  
if __name__ == '__main__':
    im = cv2.imread(image)
    #final_array = NeighborhoodAvging()
    
    #openCV built-in homogeneous blur
    #final_array = cv2.blur(np.asarray(cv2.imread(image)),(600,600))    
    
    #Guassian blur
    #final_array = cv2.GaussianBlur(np.asarray(cv2.imread(image)), (51,51), 1)
    #final_array = np.asarray(cv2.imread(image))
    
    #Binary Thresholding with guassian blur pre-processing
    #ret, final_array = binThresholdAndGuassianBlur(image) 
    
    finalImage = adaptiveThreshold(image)
    #otsuThreshold(image)
    im = np.asarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    im = cv2.GaussianBlur(im, (75,75), 1)
    finalImage = cv2.Laplacian(im, cv2.CV_32FC4)
    
#    ret ,finalImage = binThresholdAndGuassianBlur(im)
#    print type(finalImage)
    showImage(im, finalImage)

    


    
