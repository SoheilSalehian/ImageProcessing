from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.shortcuts     import buildNetwork

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import rand
import random
sampleSize = 100
imageSize = 20*20

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




if __name__ == '__main__':
    
    data = GenerateData()
    data.GenerateFinaldata()
    a = np.reshape(data.x[3], (20,20))
  
    cv2.imshow('image',a)
    cv2.waitKey(3000)
    
    #use 60% of the dataset as training set with corresponding desired output set
#    trainingSet = np.hstack((data.x[0:sampleSize*0.6], data.y[0:sampleSize*0.6]))
#    print trainingSet.shape

    # Create the dataset based on input matrix x and output vector y in numpy form
    dataSet = np.zeros((data.x[0:sampleSize].shape[0], data.x[0:sampleSize].shape[1]+1))
    dataSet[:,0:imageSize] = data.x[0:sampleSize]
    dataSet[:,-1] = data.y[0:sampleSize]
    
    # Transform the dataset into a pybrain classification dataset ("input", "target")
    x = dataSet[3]
    #print len(list(x[0:imageSize]))
    # Data set classification configuration
    ds = ClassificationDataSet(imageSize, 1, nb_classes= 2, class_labels=['Tank', 'Castle'])
    for x in dataSet:
        #all but last column is input, last element is the desired output aka. "target"
        ds.addSample(list(x[0:imageSize]), x[-1])
    # Split the data into 60% training set and 40% testing set
    testSet, trainingSet = ds.splitWithProportion(0.4)
    # One-vs-all output scheme suitable for classification on both sets
    trainingSet._convertToOneOfMany()
    testSet._convertToOneOfMany()
    # Debug printing
    print "Number of training patterns:", len(trainingSet)
    
    
     
    #create the feedforward network with 5->3->2 connection
    n = FeedForwardNetwork()
    inLayer = LinearLayer(imageSize)
    hiddenLayer = SigmoidLayer(35) 
    outLayer = LinearLayer(2)
    #add to the proper modules network
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    #Fully connect all layers
    inHiddenLayer = FullConnection(inLayer, hiddenLayer)
    hiddenOutLayer = FullConnection(hiddenLayer, outLayer)
    #add the connections to the network
    n.addConnection(inHiddenLayer)
    n.addConnection(hiddenOutLayer)
    #internal net initialization
    n.sortModules()
#    print n.activate(trainingSet[3])

    
    # Shortcut build method of the ANN
#    n = buildNetwork( trainingSet.indim, 5, trainingSet.outdim, outclass=SoftmaxLayer)
    # Construct the trainer object (using backprop)
    trainer = BackpropTrainer(n, dataset=trainingSet, learningrate=0.01, verbose=True, weightdecay=0.00)
    # Start the training iterations
    
    for i in xrange(30):
        #one epoche or pattern at a time
        trainer.trainEpochs(5)
        #classification of one-vs-all with precent error
        trainResult = percentError( trainer.testOnClassData(), trainingSet['class'] )
        testResult = percentError( trainer.testOnClassData(dataset=testSet), testSet['class'] )
        print "Pattern: %4d" % trainer.totalepochs
        print " train error: %5.2f%%" % trainResult
        print " test error: %5.2f%%" % testResult
   
            


#    

#    n.sortModules()
#    print data.x[0:sampleSize*0.6].shape
#    n.activate(data.x)
#    print n.in_to_hidden.params
