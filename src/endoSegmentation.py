from threshold import *
from matplotlib import pyplot as plt
import pylab

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
    #change image to YUV scale
    yuvImage = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    #seperate the channels and take the luma channel
    yImage = cv2.split(yuvImage)[0]
    uImage = cv2.split(yuvImage)[1]
    vImage = cv2.split(yuvImage)[2]
    #add the luma and first chrominance channel
    finalImage = yImage + uImage
    
    #simple inverse binary threshold
    ret, finalImage = cv2.threshold(finalImage, 135, 255, cv2.THRESH_BINARY_INV)
    #dilation with a small kernel and very few iteration to avoid complex inpainting
    finalImage = cv2.dilate(finalImage, kernel=np.ones((8,8)), iterations=1)
    #inpainting algorithm with low inpaint radius
    finalImage = cv2.inpaint(grayImage, finalImage, inpaintRadius=0.01, flags=cv2.INPAINT_TELEA)
    return finalImage, grayImage
    
    
    
    
def plotPretty(rowNumber, columnNumber, windowNumber, windowTitle, array):
    plt.subplot(rowNumber,columnNumber,windowNumber)
    plt.title(windowTitle)
    plt.imshow(array)
    

if __name__ == '__main__':

    
#    #calculate the histogram
#    yHist, bins = np.histogram(yImage, bins=256, range=[0,256], normed=True)
#    plt.plot(yHist, color='g')
#    plt.show()

    #run endo segmentation algoritm
    finalImage, grayImage = endoSegmentv1(image)
    

    #show the images and their respective fourier spectrums
#    plt.gray()
    p1 = plotPretty(2,2,1, 'Fourier Spectrum OriginalImage', np.log(np.abs(np.fft.fftshift(np.fft.fft2(grayImage))))**2)
    p2 = plotPretty(2,2,2, 'Fourier Spectrum FilteredImage', np.log(np.abs(np.fft.fftshift(np.fft.fft2(finalImage))))**2)
    p3 = plotPretty(2,2,3,'Original Image', grayImage)
    p4 = plotPretty(2,2,4,'Filtered Image', finalImage)
    plt.show()

    

    



