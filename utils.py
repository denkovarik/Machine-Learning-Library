# File: utils.py
# Author: Dennis Kovarik
# Purpose: Holds utility functions for running the linear regression and axis
#          aligned rectangles classifier.


import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def genPointsFittedToLine(m, b):
    """
    This function will generate and return numpy arrays containing generated
    data for labels and features fitted to a line/plane. The dimension of the
    returned data is determined by the length of the 'm' numpy array passed
    into the function. The numpy array 'm' contains the slopes for each
    for the generated data in each dimension. This function will return 100
    generated datapoints in the form of 2 numpy arrays 'x' and 'y'. The array
    'x' will contain all the features for the generated datapoints while the
    array 'y' will contain the labels for every x. 

    :param m: The slope of the fitted line to the generated data points
    :param b: The y intercept of the fitted line
    :return: 2D numpy array with the [x y] coordinates of the generated point
    """
    # Create array to hold points
    x = np.empty((0,len(m)))
    for n in range(100):
        feat = np.random.randint(0, 100, size=(1, len(m)))
        x = np.append(x, feat, axis=0)
    
    y = np.zeros((100,1))

    # Populate array with points
    for i in range(len(x)):
        for j in range(len(m)):
            y[i] = y[i] + int(m[j] * x[i,j]) + random.randint(-50, 50)
        y[i] = y[i] + b
    
    return x, y


def plotAxisAlignedRectangles2D(x, y, w=None, title="Scatter Plot", xLabel="X", yLabel="Y", \
                                positiveClassLabel="+1", negativeClassLabel="-1"):
    """
    This function plots the the data and axis  in 2D.

    :param x: 2D numpy array of the dataset features
    :param y: Numpy array of the labels for the dataset
    :param w: Weights for representing the axis aligned rectangle
    :param title: The title for the plot
    :param xLabel: The label for the x axis
    :param yLabel: The label for the y axis
    :param positiveClassLabel: Label for the positive class
    :param negativeClassLabel: Label for the negative class
    :returns: void
    """
    posIndices = np.where(y == 1)
    negIndices = np.where(y == -1)
    fig = plt.figure()
    plt.scatter(x[posIndices, 0], x[posIndices,1], marker='o', color='green', \
        label=positiveClassLabel)
    plt.scatter(x[negIndices, 0], x[negIndices, 1], marker='x', color='red', \
        label=negativeClassLabel)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    fig.legend(loc='lower right')

    if not w is None:
        # Plot the axis aligned rectangle
        plt.plot(w[0], np.array([w[1][0], w[1][0]]), color='blue') 
        plt.plot(w[0], np.array([w[1][1], w[1][1]]), color='blue')
        plt.plot(np.array([w[0][0], w[0][0]]), w[1], color='blue')
        plt.plot(np.array([w[0][1], w[0][1]]), w[1], color='blue')

    plt.show()
   

def plotAxisAlignedRectangles3D(x, y, w=None, title="Scatter Plot", xLabel="X", \
    yLabel="Y", zLabel="Z", positiveClassLabel="+1", negativeClassLabel="-1"):
    """
    This function plots the the data and axis  in 2D.

    :param x: 2D numpy array of the dataset features
    :param y: Numpy array of the labels for the dataset
    :param w: Weights for representing the axis aligned rectangle
    :param title: The title for the plot
    :param xLabel: The label for the x axis
    :param yLabel: The label for the y axis
    :param zLabel: The label for the z axis
    :param positiveClassLabel: Label for the positive class
    :param negativeClassLabel: Label for the negative class
    :returns: void
    """
    # Make sure z represents a plane 
    if not w is None and len(w) != 3:
        print("w must have length of 3 for displaying regression line in 3D")
        return

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Plot Scatter 
    pos = np.where(y == 1)
    neg = np.where(y == -1)
    plt.title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    ax.scatter(x[pos,0], x[pos,1], x[pos,2], c='green', marker='o', alpha=1, \
        label=positiveClassLabel)
    ax.scatter(x[neg,0], x[neg,1], x[neg,2], c='red', marker='x', alpha=1, \
        label=negativeClassLabel)
    fig.legend(loc='lower right')

    # Plot Axis aligned rectangle
    if not w is None:
        xs = np.linspace(w[0][0], w[0][1], 10)
        zs = np.linspace(w[2][0], w[2][1], 10)
        X, Z = np.meshgrid(xs, zs)
        Y = X - X + w[1][1]
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)

        xs = np.linspace(w[0][0], w[0][1], 10)
        zs = np.linspace(w[2][0], w[2][1], 10)
        X, Z = np.meshgrid(xs, zs)
        Y = X - X + w[1][0]
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)

        ys = np.linspace(w[1][0], w[1][1], 10)
        zs = np.linspace(w[2][0], w[2][1], 10)
        Y, Z = np.meshgrid(ys, zs)
        X = Y - Y + w[0][1]
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)

        ys = np.linspace(w[1][0], w[1][1], 10)
        zs = np.linspace(w[2][0], w[2][1], 10)
        Y, Z = np.meshgrid(ys, zs)
        X = Y - Y + w[0][0]
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)

        xs = np.linspace(w[0][0], w[0][1], 10)
        ys = np.linspace(w[1][0], w[1][1], 10)
        X, Y = np.meshgrid(xs, ys)
        Z = Y - Y + w[2][0]
        ax.plot_surface(X, Y, Z, color='yellow', alpha=0.3)

        xs = np.linspace(w[0][0], w[0][1], 10)
        ys = np.linspace(w[1][0], w[1][1], 10)
        X, Y = np.meshgrid(xs, ys)
        Z = Y - Y + w[2][1]
        ax.plot_surface(X, Y, Z, color='yellow', alpha=0.3)

    plt.show() 


def plotLogisticRegressionDecisionRegions(X, y, classifier, \
    title="Decision Boundary", xLabel="x", yLabel="y", \
    class1="class1", class2="class2", resolution=0.02):
    """
    This plots the decision boundary determined by the LogisticRegression class

    :param self: Instances of the LogisticRegression class.
    :param X: Numpy array containing the dataset attributes
    :param y: Numpy array containing the dataset labels
    :param title: The title for the plot
    :param xLabel: Label for x axis
    :param yLabel: Label for y axis
    :param class1: Label for class 1
    :param class2: Label for class 2
    :param resolution: The resolution for the graph
    :return: none
    """
    # setup marker generator and color map 
    markers = ('o', 'x', 'o', '^', 'v') 
    colors = np.array(['red', 'blue', 'lightgreen', 'gray', 'cyan'])
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
    np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predictClass(np.array([xx1.ravel(), xx2.ravel()]).T) 
    Z = Z.reshape(xx1.shape) 
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    # plot class samples 
    for idx, cl in enumerate(np.unique(y)):
        # Determine class label
        if cl <= 0.000001:
            classLabel = class1
        else:
            classLabel = class2
        theColor = colors[idx] 
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
        alpha=0.8, c=np.array([str(theColor)]), 
        marker=markers[idx], label=classLabel)

    plt.legend(loc='upper left')
    plt.show()


def plotPerceptronDecisionRegions(X, y, classifier, title="Decision Boundary", \
    xLabel="x", yLabel="y", class1="class1", class2="class2", resolution=0.02):
    """
    This plots the decision boundary determined by the perceptron

    :param self: Instances of the Perceptron class.
    :param x: Numpy array containing the dataset attributes
    :param y: Numpy array containing the dataset labels
    :param title: The title for the plot
    :param xLabel: Label for x axis
    :param yLabel: Label for y axis
    :param class1: Label for class 1
    :param class2: Label for class 2
    :param resolution: The resolution for the graph
    :return: none
    """
    # setup marker generator and color map 
    markers = ('o', 'x', 'o', '^', 'v') 
    colors = np.array(['red', 'blue', 'lightgreen', 'gray', 'cyan'])
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
    np.arange(x2_min, x2_max, resolution)) 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) 
    Z = Z.reshape(xx1.shape) 
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    # plot class samples 
    for idx, cl in enumerate(np.unique(y)):
        # Determine class label
        if cl == -1:
            classLabel = class1
        else:
            classLabel = class2
        theColor = colors[idx] 
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], 
        alpha=0.8, c=np.array([str(theColor)]), 
        marker=markers[idx], label=classLabel)

    plt.legend(loc='upper left')
    plt.show()


def plotRegression(x, y, w=None, title="Scatter Plot", xLabel="x", yLabel="y", \
                   zLabel=None):
    """
    This function plots the regression line in 2D or 3D for multivariant 
    regression that has 1 or 2 independent variables respectively.

    :param x: 2D numpy array of the dataset features
    :param y: Numpy array of the labels for the dataset
    :param w: Weights for representing the regression line for the fitted model.
    :param title: The title for the plot
    :param xLabel: The label for the x axis
    :param yLabel: The label for the y axis
    :param zLabel: The label for the z axis
    """
    # Check that x is not empty
    if len(x) <= 0:
        print("Dataset empty")
        return
    # Check if 1 independent variable
    if len(x[0]) == 1:
        # Plot the regression in 2D
        plotRegression2D(x, y, w, title, xLabel, yLabel)
        return
    if len(x[0]) == 2:
        plotRegression3D(x, y, w, title, xLabel, yLabel, zLabel)
        return
    return      # Otherwise just return
   

def plotRegression2D(x, y, w=None, title="Scatter Plot", xLabel="x", yLabel="y"):
    """
    This function plots the regression line in 3D for multivariant regression
    that has 1 independent variables.

    :param x: 2D numpy array of the dataset features
    :param y: Numpy array of the labels for the dataset
    :param w: Weights for representing the coefficients for the line regression 
              line for the fitted model.
    :param title: The title for the plot
    :param xLabel: The label for the x axis
    :param yLabel: The label for the y axis
    :param zLabel: The label for the z axis
    """
    # Plot the data
    fig = plt.figure()
    plt.scatter(x[:], y[:], color='red')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    if not w is None:
        # Plot Regression line
        xMin = np.min(x)
        xMax = np.max(x)
        regLineX = np.array([xMin, xMax])
        regLineY = np.array([xMin * w[1] + w[0], xMax * w[1] + w[0]])
        regression = plt.plot(regLineX, regLineY)
        fig.legend(regression, ["Regression Line"], loc='lower right')
    plt.show()


def plotRegression3D(x, y, w=None, title="Scatter Plot", xLabel="x", yLabel="y",\
                   zLabel="z"):
    """
    This function plots the regression line in 3D for multivariant regression
    that has 2 independent variables.

    :param x: 2D numpy array of the dataset features
    :param y: Numpy array of the labels for the dataset
    :param w: Weights for representing the regression plane for the fitted model. 
              This can only be a plane in 2D because regression for higher 
              dimensions can not be 
              easily visualized.
    :param title: The title for the plot
    :param xLabel: The label for the x axis
    :param yLabel: The label for the y axis
    :param zLabel: The label for the z axis
    """
    # Make sure z represents a plane 
    if not w is None and len(w) != 3:
            print("w must have length of 3 for displaying regression line in 3D")
            return

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plt.title(title)

    if not w is None:
        # Find the min, max, and step size for data plot
        xMin = int(x[:,0].min())
        xMax = int(x[:,0].max()) + 1
        yMin = int(x[:,1].min())
        yMax = int(x[:,1].max()) + 1
        xStep = float(xMax - xMin) / 10.00
        yStep = float(yMax - yMin) / 10.00

        # Create Dimensinos
        xa = np.arange(xMin, xMax, xStep)
        ya = np.arange(yMin, yMax, yStep)
        xx, yy = np.meshgrid(xa, ya)
        z = np.zeros((10,10))

        # Init z dimension
        for i in range(len(z)):
            for j in range(len(z[0])):
                yMap = yMin + (i * yStep)       
                xMap = xMin + (j * xStep)       
                z[i,j] = w[1] * xMap + w[2] * yMap + w[0]
    
        # Plot surface
        ax.plot_surface(xx,yy,z,alpha=0.5)

    # Plot Scatter
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    ax.scatter(x[:,0], x[:,1], y[:], c='red', marker='o', alpha=0.5)
    plt.show()


def randomizeData(y, dist):
    """
    This function simply randomly displaces the points the dataset to add 
    randomness.

    :param y: Numpy array of the labels for the dataset
    :param dis: Float between 0 and 1 representing the percentage of 
                displacement to apply for each point. 
    :return: 2D numpy array of y with randomly displaced values
    """
    # Find the max and min value
    theMax = y.max()
    theMin = y.min()
    theDist = (theMax - theMin) * dist
    # Populate array with points
    for i in range(len(y)):
        y[i] = y[i] + random.uniform(-1*theDist, theDist)

    return y
