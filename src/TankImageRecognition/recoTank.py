import cv2
import numpy as np
import pybrain
from matplotlib import pyplot as plt
from pylab import rand
import random
sampleSize = 1000

class GenerateData:
    #contruct a 20*20 input matrix and the desired output vector depending on the sample size
    def __init__(self):
        self.x = np.zeros((sampleSize ,20*20))
        self.y = np.zeros(sampleSize)
    
    #Generate a simple Tank on the 20*20 matrix with the ability to offset in the vertical/horizontal directions
    def GenerateTankPixel(self, xOffset, yOffset):
        a = np.zeros((20,20))
        a[10+xOffset:16+xOffset,8+yOffset] = 1
        a[10+xOffset:16+xOffset,14+yOffset] = 1
        a[12+xOffset:14+xOffset,9+yOffset:14+yOffset] = 1
        a[6+xOffset:12+xOffset,11+yOffset] = 1
        a = a.ravel()
        return a
    
    #Generate a simple castle on the 20*20 matrix with the ability to offset in the vertical/horizontal directions
    def GenerateCastlePixel(self, xOffset, yOffset):
        a = np.zeros((20,20))
        a[10+xOffset:13+xOffset,8+yOffset] = 1
        a[10+xOffset:13+xOffset,16+yOffset] = 1
        a[10+xOffset:13+xOffset,8+yOffset:16+yOffset] = 1
        a[6+xOffset:13+xOffset,12+yOffset] = 1
        a[5+xOffset:13+xOffset,11+yOffset] = 1
        a[5+xOffset:13+xOffset,13+yOffset] = 1
        a[8+xOffset,12+yOffset] = 0
        a = a.ravel()
        return a
    
    #randomly generates castles and tanks in different directions with rotation variance introduced
    def GenerateFinaldata(self):
        for i in xrange(sampleSize):
            castleFlag = random.randint(0,2)
            if (castleFlag):
                self.x[i,]=self.GenerateCastlePixel(random.randint(-2,8), random.randint(-7,3))
                #desired output is a non-tank(0)
                self.y[i] = 0
            else:
                self.x[i,]=self.GenerateTankPixel(random.randint(-2,7), random.randint(-7,5))
                #desired output is a tank(1)
                self.y[i] = 1
            #rotate
            self.x[i,] = self.rotationVariance(self.x[i,],i)
                
                
    #introduces rotation variance to the images in a random fashion
    def rotationVariance(self, x, i):
        for j in range(random.randint(0,3)):
            b=np.reshape(self.x[i], (20,20))
            #horizontal rotation
            b = b[::-1]
            #vertical rotation
            b = b.T
            self.x[i,] = b.ravel()
#            print "shape is", b.shape
        return self.x[i,]      

                    
                

desiredResponse = np.zeros((20,20))

class Neuron:
    #random initialization to break symmetry problem
    def __init__(self):
        self.w = np.zeros((sampleSize,20*20))
        self.b = 1
        self.learningRate = 0.001
        self.response = 0
    
    #the output response    
    def response(self, x):
        #actual response
        #print "x", x.shape
        #print "w", self.w.shape
        actualResponse = int(np.dot(self.w, x))
        print "actual response:",actualResponse
        #simple step activation function 
        self.response = 1/1+math.exp(actualResponse)
    
    #update the weights at time t+1: w(t+1) = w(t) + learningRate * (yd - ya), where yd= desired output and ya= actual output iterError = yd - ya
    def updateW(self, sample, iterError, i):
        print self.w[i].shape
#        print "w before", self.w[0,temp]
        self.w[i] += np.dot(self.learningRate*iterError,sample)
#        print "w after", self.w[0,temp]
        self.b += iterError
    
    #training based on the data passed on
    def train(self,data):
        learned = False
        iteration = 0
        i = 0
        while not learned:
            globalError = 0.0
            for sample in data:
                actualResponse = self.response(sample)
                #if the actual response is not the same as its desired response
                if actualResponse != desiredResponse[i]:
                    #find the error
                    iterError = desiredResponse[i] - actualResponse
                    #update the weights based on the error
                    self.updateW(sample, iterError, i)
                    #track the global error
                    globalError += abs(iterError)
                    print "Iteration:", iteration, "b:", self.b, "Iteration Error:", iterError
                    print "desired:", desiredResponse[i], "vs. actual:", actualResponse
                iteration +=1
                #increment desired output index to go to the next 
                i += 1
                
            #termination criteria
            if globalError == 0.0 or iteration >= 600:
                print "Final Iteration", iteration
                #terminate learning
                learned = True
                

class NeuralNet(Neuron):
    def __init__(self):
        for 
        self.hPerceptron = Neuron()
        self.vPerceptron = Perceptron()
        self.finalPerceptron = Perceptron()
        
    def finalResponse(self,x):
        #response of the final perceptron
        actualResponse = self.hPerceptron.response(x)*self.finalPerceptron.w[0] + self.vPerceptron.response(x)*self.finalPerceptron.w[1]
        #simple step activation function 
        if actualResponse >= 0:
            return 1
        else:
            return 0
        
                
                           
        

if __name__ == '__main__':
    
    data = GenerateData()
    data.GenerateFinaldata()
#    print data.y.shape
    a = np.reshape(data.x[3], (20,20))
#    print data.y[3]
    
    
    cv2.imshow('image',a)
    cv2.waitKey(3000)
    

    twoLayer = NeuralNet()
    
    
    #use 60% of the dataset as training set with corresponding desired output set
    desiredResponse = data.y[0:sampleSize*0.6]
    myPerceptron.train(data.x[0:sampleSize*0.6])
    
    #the other 40% of the data set is stored as the testing set with its corresponding output set
    testSet = data.x[sampleSize*0.6+1:]
    desiredTestResponse = data.y[sampleSize*0.6+1:]
    
    
    
    
#    a = np.((3,1))
#    b = np.ones((1,3))
#    print a[:,-1]
#    