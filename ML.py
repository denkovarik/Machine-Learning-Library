# File: ML.py
# Author: Dennis Kovarik
# Purpose: Holds Machine Learning Modules

import cvxopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv
from math import sqrt
import math
import random


class AxisAlignedRectangles:
    def __init__(self):
        """
        This function initializes the AxisAlignedRectangles class.

        :param self: Instance of the AxisAlignedRectangles class.
        :return: None
        """
        self.w = np.array([])
        self.errors = np.array([])


    def error(self, x, y):
        """
        This function counts the number of misclassified instances in the 
        dataset. For each correctly classified instance, a value of 1 will be
        placed in the corresponding index of the self.errors numpy array. Each
        misclassified point will be represented with a value of -1 in the 
        corresponding index in the self.errors numpy array.

        :param self: Instance of the AxisAlignedRectangle class.
        :param x: The features for each point in the dataset.
        :param y: The class label for the data.
        :return: numpy array of the misclassified instances
        """
        # Make sure dataset is not empty
        if len(x) <= 0:
            print("Dataset can not be empty")
            return

        # Chech that x and y are of the same lengths
        if len(x) != len(y):
            print("x and y must have the same lengths")
            return

        # Check if weights were initialized correctly
        if len(self.w) != len(x[0]):
            self.initRectangle(len(x[0]))

        if len(self.errors) != len(x):
            self.errors = np.zeros(len(x))
        
        predictions = self.predict(x)
        self.errors = np.ones(len(x))
        self.errors[np.where(predictions != y)] = -1        


    def fit(self, x, y):
        """
        This function attempts to fit a line to model the data in the training
        set.

        :param self: Instance of the AxisAlignedRectangle class.
        :param x: The features for each point in the dataset.
        :param y: The class label for the data.
        :return: None
        """
        if len(x) == 0:
            print("Dataset can not be empty")
            return

        # Chech that x and y are of the same lengths
        if len(x) != len(y):
            print("x and y must have the same lengths")
            return
        
        # Make sure rectangle was initialized
        if len(x[0]) != len(self.w):
            self.initRectangle(len(x[0]))


        # Sort data for positive labels based on class labes
        pos = x[np.where(y == 1)]
        # Init the weights
        for d in range(len(x[0])):
            self.w[d,0] = np.min(pos[:,d])
            self.w[d,1] = np.max(pos[:,d])
        
        bestWeights = self.w.copy()
        predictions = self.predict(x)
        misclassified = np.where(predictions != y)
        self.error(x,y)
        lowestError = len(np.where(predictions != y)[0])

        for d in range(1,len(pos[0])):
            # Adjust min
            for p in range(len(pos)):
                if pos[p,d] <= self.w[d,1]:
                    self.w[d,0] = pos[p,d]
                    predictions = self.predict(x)
                    numMisclassified = len(np.where(predictions != y)[0])
                    if numMisclassified < lowestError:
                        bestWeights = self.w.copy()
                        lowestError = numMisclassified
            self.w = bestWeights.copy()
            # Adjust the max
            for p in range(len(pos)):
                if pos[p,d] >= self.w[d,0]:
                    self.w[d,1] = pos[p,d]
                    predictions = self.predict(x)
                    numMisclassified = len(np.where(predictions != y)[0])   
                    if numMisclassified < lowestError:
                        bestWeights = self.w.copy()
                        lowestError = numMisclassified
            self.w = bestWeights.copy() 


    def getErrors(self, x, y):
        """
        This function is a getter function for the class numpy array 'errors'.

        :param self: Instance of the AxisAlignedRectangle class.
        :param x: The features for each point in the dataset.
        :param y: The class label for the data.
        :return: A numpy array containing the correctly classified and 
                 misclassified points.
        """
        self.error(x, y)
        return self.errors.copy()
    

    def getWeights(self):
        """
        This function is a getter function for boundaries of the axis aligned 
        rectangle.

        :param self: Instance of the AxisAlignedRectangle class.
        :return: A 2D numpy array containing the boundaries for the axis 
                 aligned rectangle. 
        """
        return self.w.copy()


    def initRectangle(self, dim):
        """
        This function initializes the model's axis aligned rectangle to the 
        dimension specified by 'dim'

        :param self: Instance of the Linear Regression class.
        :param dim: The dimension for the rectangle
        :return: None
        """
        self.w = np.zeros((dim, 2))


    def predict(self, x):
        """
        This function will attempt to predict the class label of all the
        instances in x.

        :param self: Instance of the Linear Regression class.
        :param x: The features for each point in the dataset.
        :return: A 1D numpy array for the predicted class labels for every
                 instance in x
        """
        # Make sure x is not empty
        if len(x) <= 0:
            print("The dataset can not be empty")
            return np.array([])

        # Make sure rectangle was initialized
        if len(x[0]) != len(self.w):
            print("Warning, weights for model was not initialized")
            self.initRectangle(len(x[0]))

        # Init predictions numpy array
        predictions = np.ones(len(x))

        # Label instances outside of rectangle
        for i in range(len(self.w)):
            #print(self.w[i])
            predictions[np.where(x[:,i] < self.w[i,0])] = int(-1)
            predictions[np.where(x[:,i] > self.w[i,1])] = int(-1)

        return predictions
       

class KNearestNeighbors:
    def __init__(self, nn):
        """
        This function initializes the K Nearest Neighbors model.

        :param self: Instance of the K Nearest Neighbors model
        :param nn: The number of nearest neighbors to make a prediction on
        :return: None
        """
        self.numNeighbors = nn

    
    def distance(self, instance1, instance2):
        """
        Calculates the euclidean between 2 instances in the dataset. 

        :param self: Instance of the K Nearest Neighbors model.
        :param instance1: The features of an instance of the dataset
        :param instance2: The features of an instance of the dataset
        :return: The euclidean distance between the two instances
        """
        dis = 0.0
        for i in range(len(instance1)):
            dis = dis + (instance1[i] - instance2[i])**2
        dis = sqrt(dis)
        return dis


    def fit(self, x, labels):
        """
        This function fits K Nearest Neighbors model to the data passed in.
        It basically takes in the training dataset and stores it.

        :param self: Instance of the K Nearest Neighbors model
        :param x: Feature vector for the objects in the training dataset
        :param label: The class labels for each of the data objects in x
        :return: None
        """
        if len(x) != len(labels):
            print("Number on instances between x and y must be equal")
            exit()
        y = np.zeros(shape=(len(labels),1))
        for i in range(len(y)):
            y[i,0] = labels[i]
        self.trainingData = np.hstack((x, y))
        self.classLabels = np.unique(y)


    def getNeighbors(self, xTest):
        """
        This function finds and returns the k closest neighbors to the 
        data object 'xTest'. The distance is determined by measuring the
        euclidean distance between any two objects.

        :param self: Instance of the K Nearest Neighbors model
        :param xTest: Object to find the k nearest neighbors for
        :return: List of the k nearest neighbors to xTest
        """
        distances = list()
        for xInstance in self.trainingData:
            dist = self.distance(xInstance[0:(len(xInstance)-1)], xTest) 
            distances.append((xInstance, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.numNeighbors):
            neighbors.append(distances[i][0])
        return neighbors


    def predictClass(self, instance):
        """
        This function predicts the class label of a data object by observing 
        its k nearest neighbors. The predicted class label will be the class
        that is held by the majority of the data objects k closest neighbors. 
        For ties, one of the labels that are tied for the max will be randomly
        selected for the predicted class of 'instance'.

        :param self: Instance of the K Nearest Neighbors model
        :param instance: Object to predict the class label for.
        :return: The predicted class label for the data object
        """
        neighbors = self.getNeighbors(instance)
        maxNumClassInstances = 0
        maxClass = 0
        labelI = len(neighbors[0]) - 1
        counts = np.zeros(len(self.classLabels))
        for i in range(len(self.classLabels)):
            for j in range(len(neighbors)):
                if self.classLabels[i] == neighbors[j][labelI]:
                    counts[i] += 1
        maxIndices = np.where(counts == np.amax(counts))[0]
        return self.classLabels[maxIndices[random.randint(0, len(maxIndices)-1)]]


    def predict(self, tests):
        """
        This function predicts the class labels for all object contained in a 
        numpy array. It does this by calling the KNearestNeighbors 
        'predictClass()' function.

        :param self: Instance of the K Nearest Neighbors model
        :param tests: Array of data objects to predict class labels for 
        :return: A numpy array for the predicted classes for each object in tests
        """
        testLen = len(tests)
        preds = np.zeros(shape=(testLen))
        for i in range(testLen):
            preds[i] = self.predictClass(tests[i])
        return preds


class LinearRegression:
    def __init__(self):
        """
        This function initializes the Linear Regression model.

        :param self: Instance of the LinearRegression class.
        :return: None
        """
        self.w = np.array([])


    def fit(self, x, y):
        """
        This function attempts to fit a line to model the data in the training
        set.

        :param self: Instance of the Linear Regression class.
        :param x: The features for each point in the dataset
        :param y: The true values for each point in the dataset
        :return: None
        """
        if len(x) == 0:
            print("Dataset can not be empty")
            return
 
        m = len(x)
        X = x.copy()
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        
        b = np.zeros(X.shape[1]) #initialize the value of coefficients as 0.
        pred = X.dot(b)
        cost = np.mean(np.square(X.dot(b) - y)) / 2.0

        # Books Implementation
        self.w = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))
        
        pred = X.dot(self.w)
        cost = np.mean(np.square(X.dot(self.w) - y)) / 2.0
        #print(cost)


    def getWeights(self):
        """
        This function returns the weights for the model

        :param self: Instance of the Linear Regression class.
        :return: The weights for the model as a 2D numpy array.
        """
        return self.w

    
    def predict(self, x):
        """
        This function predicts the output value for the attributes held in the 
        numpay array x.

        :param self: Instance of the Linear Regression class.
        :param x: Numpy array holding the attributes for the dataset
        :return: The predictions of the dataset
        """
        if len(x) == 0:
            print("Dataset can not be empty")
            return -1

        if len(x[0]) != len(self.w):
            print("Warning, model was not initailized to dataset")
            # Init weights for the model
            self.w = np.random.randn(1, len(x[0]))    
            
        predictions = np.zeros(len(x)) 

        # Make prediction
        for j in range(len(x)):
            predictions[j] = np.dot(self.w, x[j]) + self.b
            
        return predictions


    def squaredError(self, x, y):
        """
        This function computes the squared error between the datasets and the 
        regression line determined by the model's weights.

        :param self: Instance of the AxisAlignedRectangle class.
        :param x: The features for each point in the dataset.
        :param y: The class label for the data.
        :return: The squared error as a float
        """
        if len(x) == 0:
            print("Dataset can not be empty")
            return -1

        if len(x[0]) + 1 != len(self.w):
            print("Warning, model was not initailized to dataset")
            self.w = np.zeros(len(x[0])+1)    # Init weights for the model

        X = x.copy()
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X)) 
        pred = X.dot(self.w)
        cost = np.mean(np.square(X.dot(self.w) - y)) / 2.0
        return cost
    

class LogisticRegression:
    def __init__(self, alpha=0.1, epochs=30):
        """
        Initializes the logistic regression model.

        :param self: Instance of the LogisticRegression class.
        :param alpha: learning rate (default:0.01)
        :param epochs: maximum number of iterations of the logistic regression 
                       algorithm for a single run (default=30)
        :return: None
        """
        self.alpha = alpha
        self.epochs = epochs
        self.weights = np.random.randn(3)


    def cost(self, y_estimated, y): 
        cost = np.mean(-y * np.log(y_estimated) - (1 - y) * np.log(1-y_estimated))
        return cost


    def fit(self, X, y, miniBatchSize=3):
        """
        :param self: Instance of the LogisticRegression class.
        :param X: Dataset containing data object features
        :param y: Class labeles for X
        :param miniBatchSize: The size of the minibatches for stochastic gradient 
                          descent.
        :return: list of the cost function changing overtime
        """
        # Get the total number of samples
        totNumSamples = np.shape(X)[0]
        numFeatures = np.shape(X)[1]
        X = np.concatenate((np.ones((totNumSamples, 1)), X), axis=1)
        # Init the weights
        self.weights = np.random.randn(numFeatures + 1, )
        # stores the updates on the cost function (loss function)
        trainingLoss = []
        s = np.arange(X.shape[0])
        minStepCount = 0

        # Stochastic Gradient Descent
        for epoch in range(self.epochs):
            # Randomly shuffle X and y before splitting into minibatches
            np.random.shuffle(s)
            # Split into minibatches
            numSplits = math.ceil(len(X) / miniBatchSize)
            minBatchX = np.array_split(X[s], numSplits)
            minBatchY = np.array_split(y[s], numSplits)
            cost = 0    # Init the cost
            stop = True

            for i in range(len(minBatchX)):
                # Make a prediction
                pred = self.predict(minBatchX[i])
                # calculate the difference between the actual and predicted value
                error = (pred - minBatchY[i])
                # find the cost
                cost += self.cost(pred, minBatchY[i])
                # Find the gradient
                gradient = (1 / totNumSamples) * minBatchX[i].T.dot(error)
                stepSize = self.alpha * gradient    
                # Update our weights
                self.weights = self.weights - stepSize
                # Stoping Criteria
                if np.absolute(stepSize).max() > 0.001:
                    stop = False

            # Stop when the absolute value of the stepSize < 0.001
            if stop:
                return self.weights, trainingLoss
            # Record the training loss for the epoch
            trainingLoss.append(cost)

        return self.weights, trainingLoss


    def predict(self, X):
        """
        This function predicts the class labels of the data objects passed 
        into it.
        """
        return self.sigmoid(X.dot(self.weights))

    
    def predictClass(self, X):
        X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
        prob = self.sigmoid(X.dot(self.weights))
        return np.round(prob)


    def sigmoid(self, x):
        """
        :param x: input value
        :return: the sigmoid activation value for a given input value
        """
        return 1 / (1 + np.exp(-x))


class Perceptron:
    def __init__(self, learningRate=None, numIterations=None):
        """
        This function initializes the Perceptron object and checks the input
        parameters.

        :param self: Instance of the Perceptron class.
        :param learningRate: Float of the learning rate for the Perceptron
        :param numIterations: Max iterations the perceptron will train for
        :return: None
        """
        # init learing rate and number of iterations to defaults
        self.rate = 0.1
        self.niter = 10
        # init the errors array to something so it exists
        self.errors = np.array([0])
        # init the weight array to something so ti exists
        self.weight = np.array([0,0])

        # Check the type of learningRate passed in
        if type(self.rate) == type(learningRate):
            self.rate = learningRate 
        else:
            errmsg = "Error in Perceptron.__init__ ... "
            errmsg = errmsg + "value for learning rate must be of type float. "
            errmsg = errmsg + "Default value of 0.1 being used"
            print(errmsg)
        # Check type of the number of iterations passed int
        if type(self.niter) == type(numIterations):
            self.niter = numIterations
        else:
            errmsg = "Error in Perceptron.__init__ ... "
            errmsg = errmsg + "value for number of iterations must be of type "
            errmsg = errmsg + "in"
            errmsg = errmsg + "Default value of 10 being used"
            print(errmsg)

    def dotProduct(self, x):
        """
        This function computes the dot product of x with the perceptron weight 
        array. 

        :param self: Instance of the Perceptron class.
        :param x: Numpy array containing the dataset attributes
        :return: float of the dot product for x with the perceptron weight
                 array plus the bias.
        """
        # Check that the sizes of weight and x are the same
        if len(x) + 1 != len(self.weight):
            print("Sizes of x and weight arrays are not the same")
            return float('NaN')

        y = 0.0
        # Iterate through the weights
        for i in range(len(x)):
            y = y + (x[i] * self.weight[i])
        # Include the bias node
        y = y + self.weight[len(self.weight)-1]
        return y


    def fit(self, x, y):
        """
        This function trains the Perceptron on the input data x and y for 
        classiciation.

        :param self: Instance of the Perceptron class.
        :param x: Numpy array containing the dataset attributes
        :param y: Numpy array containing the class labels for x
        :return: None
        """
        # Check the type of x
        if type(x) != type(np.array([])):
            print("Error in Perceptron.fit ... x must be of type numpy array")
            return np.array(["x must be of type numpy array"])

        # Check the type of y
        if type(y) != type(np.array([])):
            print("Error in Perceptron.fit ... y must be of type numpy array")
            return np.array(["y must be of type numpy array"])
    
        # Check the dimensions of x
        if x.ndim != 2:
            print("Error in Perceptron.fit ... x must have 2 dimensions")
            return np.array(["x must have 2 dimensions"])
    
        # Check the dimensions of y
        if y.ndim != 1:
            print("Error in Perceptron.fit ... y must have 1 dimensions")
            return np.array(["y must have 1 dimension"])

        # Check that the sizes of x and y match
        if len(x) != len(y):
            print("Error in Perceptron.fit ... sizes of x and y must match. Quitting")
            return np.array(["y must have 1 dimension"])
                
        # Init the size of the weight array to num attributes plus 1 for bias
        self.weight = np.zeros(len(x[0]) + 1)
        self.errors = np.zeros(self.niter)

        for it in range(self.niter):
            #plot_decision_regions(x, y, self) 
            # Iterate through training instances
            theError = int(0);
            for j in range(len(x)):
                # Compute the dot product
                yj = self.dotProduct(x[j])

                if yj > 0:
                    yj = 1
                else:
                    yj = -1                

                if y[j] != yj:
                    theError += 1
                    # Iterate through the weights
                    for i in range(len(x[j])):
                        self.weight[i] = self.weight[i] + y[j] * self.rate * x[j][i]
                        #self.weight[i] = self.weight[i] + self.rate * (y[j]-yj) * x[j][i]
                    # Update the bias
                    bias = len(self.weight)-1
                    self.weight[bias] = self.weight[bias] + self.rate * y[j]
                    #self.weight[bias] = self.weight[bias] + self.rate * (y[j]-yj) * 1.0
            
            self.errors[it] = int(theError)
            # Return if error is zero
            if self.errors[it] == 0 and it < self.niter:
                self.errors.resize(it + 1)
                return
    

    def net_input(self, x):
        """
        This function outputs the weighted sum for all objects in x

        :param self: Instance of the Perceptron class.
        :param x: Numpy array containing the dataset attributes
        :return: a numpy array containing the weighted sums for all objects in x
        """
        predictions = np.zeros(len(x))
        for j in range(len(x)):
            predictions[j] = self.dotProduct(x[j])

        return predictions
      

    def predict(self, x):
        """
        This function predicts the class label for the attributes held in the 
        numpay array x.

        :param self: Instance of the Perceptron class.
        :param x: Numpy array holding the attributes for 1 dataset instance
        :return: The class label for the dataset instance as an int
        """
        threshold = 0.0;
        predictions = np.zeros(len(x)) 
        # Make prediction
        for j in range(len(x)):
            yj = 0.0
            # Compute the dot product
            yj = self.dotProduct(x[j])
            if yj > threshold:
                predictions[j] = 1
            else: 
                predictions[j] = -1
            
        return predictions
