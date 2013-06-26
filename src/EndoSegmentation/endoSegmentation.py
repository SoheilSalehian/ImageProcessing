from threshold import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pylab

image = 'ColposcopyImages/1/2.jpg'
image = 'endo.jpg'



def fftHighPass(image, r):
#    #have image in 2-D
#    image = np.mean(image,2)
    #convert to frequency domain
    fftImage = np.fft.fft2(image)
    #center to zero frequency component
    fftImage = np.fft.fftshift(fftImage)
    #fft filtered spectrum of Image
    fftFilteredImage = np.zeros_like(fftImage, dtype=complex)
    #spectrum radius
    r = int((r*min(image.shape))/2);
    # spectrum center
    c1 = fftImage.shape[0]/2 
    c2 = fftImage.shape[1]/2
    #filtering stage
    for i in xrange(c1-r, c1+r):
        #from the center
        for j in range(c2-r, c2+r):
            #keep the values in those ranges
            fftFilteredImage[i,j] = fftImage[i,j]
    #convert back to time domain as a real number
    finalImage = np.real(np.fft.ifft2(np.fft.ifftshift(fftFilteredImage)))
    return finalImage, fftFilteredImage
#    finalImage = cv2.inpaint(finalImage, inpaintMask, inpaintRadius, flags=cv2.INPAINT_NS)


def endoSegmentv1(image):
    #read Image and apply guassian filter
    im = cv2.imread(image)
    #change image to gray scale
    grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #change image to YUV scalepoints
    yuvImage = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    #seperate the channels and take the luma channel
    yImage = cv2.split(yuvImage)[0]
    uImage = cv2.split(yuvImage)[1]
    vImage = cv2.split(yuvImage)[2]
    #add the luma and first chrominance channel
    finalImage = yImage + uImage
    return finalImage, im

def contArea(im, threshImage):
    print im.size
    contourImage = np.copy(threshImage)
    temp = np.copy(threshImage)
    contours, hierachy = cv2.findContours(contourImage, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)      
    contList = []
    badList = []
    for i in xrange(len(contours)):
        if(math.fabs((cv2.arcLength(contours[i], True)) <= im.size/8000)):
#        if((cv2.contourArea(contours[i])) > im.size/200):
            contList.append(contours[i])
        elif(math.fabs((cv2.arcLength(contours[i], True)) > im.size/8000)): #and (cv2.arcLength(contours[i], True) <= 1000000000000000))):
            badList.append(contours[i])
    
    print len(badList), len(contours), len(contList)

    cv2.drawContours(threshImage, contList, -1, (0,255,0), -1)
    cv2.drawContours(temp, badList, -1, (0,255,0), -1) 
    return temp

def endoSegmentv2(image):
    #read Image 
    im = plt.imread(image)
    #take image green channel
    blueImage = cv2.split(im)[0]
    greenImage = cv2.split(im)[1]
    redImage = cv2.split(im)[2]
    
    #work with the green image
    finalImage = greenImage
    #global thresholding
    ret, finalImage = cv2.threshold(finalImage, 170, 255, cv2.THRESH_BINARY)
    
    finalImage = contArea(greenImage, finalImage)
    tempImage = np.copy(finalImage)
#    tempImage = np.copy(finalImage)
    dilateKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    finalImage = cv2.dilate(finalImage, kernel=dilateKernel, iterations=7)
#    finalImage = cv2.inpaint(im, finalImage, inpaintRadius=10, flags=cv2.INPAINT_TELEA)  
    return finalImage, tempImage

     
def plotPretty(rowNumber, columnNumber, windowNumber, windowTitle, array):
    plt.subplot(rowNumber,columnNumber,windowNumber)
    plt.title(windowTitle)
    plt.imshow(array)
    

if __name__ == '__main__':

    im = plt.imread(image)
    #run endo segmentation algoritm8
    finalImage, grayImage = endoSegmentv1(image)
    
#    #show the images 
    plt.gray()
    
    plt.subplot(1,2,1)
    plt.title("a) Imagen Original")
    plt.imshow(im, origin='lower')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title("b) Imagen Final del Algoritmo")
    plt.imshow(finalImage, origin='upper')
    plt.axis('off')

    plt.colors()
    plt.show()

    

    



