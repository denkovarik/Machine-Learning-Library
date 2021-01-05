# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ML import KNearestNeighbors
from matplotlib.colors import ListedColormap


def plotPedictionResults(x, y, correctX, correctY, misLabeledX, mislabeledy):
    """
    This function helps to plot the prediction results for the K Nearest 
    Neighbors Model.

    :param x: Feature vector for the dataset
    :param y: Class labels for the elements from x
    :param correctX: Feature vector of elements from x that were correctly 
                     classified
    :param correctY: Class labels for elements from the 'correctX' 
                     feature vector
    :param misLabeledX: Feature vector of elements from x that were 
                        misclassified
    :param misLabeledY: Class labels for elements in 'misLabeledX'
    :return: None
    """
    # Visualize Prediction Results
    plt.scatter(x[np.where(y[:] == 0), 0], x[np.where(y[:] == 0), 1], \
        color='black', marker='o', label='Iris-setosa')
    plt.scatter(correctX[np.where(correctY[:] == 0), 0], correctX[np.where(correctY[:] == 0), 1], \
        color='green', marker='o')
    plt.scatter(mislabeledX[np.where(mislabeledY[:] == 0), 0], mislabeledX[np.where(mislabeledY[:] == 0), 1], \
        color='red', marker='o')

    plt.scatter(x[np.where(y[:] == 1), 0], x[np.where(y[:] == 1), 1], \
        color='black', marker='x', label='Iris-versicolor')
    plt.scatter(correctX[np.where(correctY[:] == 1), 0], correctX[np.where(correctY[:] == 1), 1], \
        color='green', marker='x')
    plt.scatter(mislabeledX[np.where(mislabeledY[:] == 1), 0], mislabeledX[np.where(mislabeledY[:] == 1), 1], \
        color='red', marker='x')

    plt.scatter(x[np.where(y[:] == 2), 0], x[np.where(y[:] == 2), 1], \
        color='black', marker='^', label='Iris-virginica')
    plt.scatter(correctX[np.where(correctY[:] == 2), 0], correctX[np.where(correctY[:] == 2), 1], \
        color='green', marker='^')
    plt.scatter(mislabeledX[np.where(mislabeledY[:] == 2), 0], mislabeledX[np.where(mislabeledY[:] == 2), 1], \
        color='red', marker='^')



print("Running K Nearest Neighbors Demo")

print("\tClassifying Iris Setosa, Iris Versicolor, and Iris Virginica in the Iris Dataset using Sepal Length and Petal Width")

# Download and read the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Split data into training and testing sets
dfTrain                = df.iloc[0:45,    :]
dfTest                 = df.iloc[45:50,   :]
dfTrain = dfTrain.append(df.iloc[50:95,   :])
dfTest   = dfTest.append(df.iloc[95:100,   :])
dfTrain = dfTrain.append(df.iloc[100:145, :])
dfTest   = dfTest.append(df.iloc[145:150, :])

# Extract class labels
yTrain = np.zeros(shape=(len(dfTrain)))
yTrain[np.where(dfTrain[:][4] == 'Iris-versicolor')] = 1
yTrain[np.where(dfTrain[:][4] == 'Iris-virginica')] = 2
yTest = np.zeros(shape=(len(dfTest)))
yTest[np.where(dfTest[:][4] == 'Iris-versicolor')] = 1
yTest[np.where(dfTest[:][4] == 'Iris-virginica')] = 2


# Extract the sepal length and pedal width
xTrain = dfTrain.iloc[:, [0, 3]].values
xTest =  dfTest.iloc[:, [0, 3]].values


# Visualize the Data
plt.scatter(xTrain[np.where(yTrain[:] == 0), 0], xTrain[np.where(yTrain[:] == 0), 1], \
    color='black', marker='o', label='Iris-setosa')
plt.scatter(xTrain[np.where(yTrain[:] == 1), 0], xTrain[np.where(yTrain[:] == 1), 1], \
    color='black', marker='x', label='Iris-versicolor')
plt.scatter(xTrain[np.where(yTrain[:] == 2), 0], xTrain[np.where(yTrain[:] == 2), 1], \
    color='black', marker='^', label='Iris-virginica')
title = "K Nearest Neighbors Demo:\n"
title += "Training Dataset for the K Nearest Neighbors Model"
plt.title(title)
plt.xlabel('sepal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()


kList = [1, 5]
for k in kList:
    print("\t\tRunning K Nearest Neighbors Model with k equal to " + str(k))
    model = KNearestNeighbors(k)
    model.fit(xTrain, yTrain)
    testPreds = model.predict(xTest)

    correctPreds = testPreds == yTest
    incorrectPreds = testPreds != yTest
    correctX = xTest[np.where(correctPreds[:])]
    correctY = yTest[np.where(correctPreds[:])]
    mislabeledX = xTest[np.where(incorrectPreds[:])]
    mislabeledY = yTest[np.where(incorrectPreds[:])]

    # Visualize Prediction Results
    plotPedictionResults(xTrain, yTrain, correctX, correctY, mislabeledX, \
         mislabeledY)


    title = "Predictions for Test Data Determined by the K Nearest Neighbors "
    title += "Model\n"
    title += "(colored shapes = Test Data), (k = " + str(k) + "),\n"
    title += "(Green = correct prediction), (red = misclassified)"
    plt.title(title)
    plt.xlabel('sepal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.show()

    
    # Determine decision boundaries
    h = .2  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. 
    x_min, x_max = xTrain[:, 0].min() - 1, xTrain[:, 0].max() + 1
    y_min, y_max = xTrain[:, 1].min() - 1, xTrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
    # Visualize Prediction Results
    plt.scatter(xTrain[np.where(yTrain[:] == 0), 0], xTrain[np.where(yTrain[:] == 0), 1], \
        color='red', marker='o', label='Iris-setosa')
    plt.scatter(xTrain[np.where(yTrain[:] == 1), 0], xTrain[np.where(yTrain[:] == 1), 1], \
        color='green', marker='x', label='Iris-versicolor')
    plt.scatter(xTrain[np.where(yTrain[:] == 2), 0], xTrain[np.where(yTrain[:] == 2), 1], \
        color='blue', marker='^', label='Iris-virginica')


    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    title = "Decision Boundaries Determined by K Nearest Neighbors Model\n(k"
    title += " = " + str(k) + ")"
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xlabel('sepal length')
    plt.ylabel('petal width')
    plt.show()
