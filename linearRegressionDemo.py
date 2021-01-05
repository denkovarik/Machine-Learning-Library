# File: linearRegressinDemo.py
# Author: Dennis Kovarik
# Purpose: Run the Linear Regression model on examples
# Usage: python3 linearRegressinDemo.py


from ML import LinearRegression
import numpy as np
from sklearn.datasets import make_regression
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd


# Linear Regression example with randomly generated data
print("Running Linear Regression Demo")
print("\tLinear Regression Example on Randomly Generated Data")
X, Y = genPointsFittedToLine(np.array([-2]), 10)
plotTitle = "Linear Regression Demo:\n"
plotTitle += "Data Visualization for Randomly Generated Data Fitted to a Line"

plotRegression(X, Y, title=plotTitle)

# Create the Model
model = LinearRegression()

# Fit the model
model.fit(X, Y)
print("\t\tSquared Error: ", end="")
print(model.squaredError(X,Y))
# Display regression plane
plotTitle = "Regression Line Determined by Linear Regression Model for the\n"
plotTitle += "Randomly Generated Data"
plotRegression(X, Y, w=model.getWeights(), title=plotTitle)



# Linear Regression example with the Iris Plants Databse
print("\tLinear Regression Example on the Iris Plants Database")
# Download and read the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Extract the first 2 features (sepal length and pedal length)
X = df.iloc[0:50, [0]].values
Y = df.iloc[0:50, [1]].values
plotTitle = "Linear Regression Demo:\n"
plotTitle += "Data Visualization for the Septal Width and Septal Length of Iris Setosa"

# Visualize the data
plotRegression(X, Y, title=plotTitle, xLabel="Sepal Length", yLabel="Sepal Width")

# Create the Model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

print("\t\tSquared Error: ", end="")
print(model.squaredError(X,Y))

# Display regression line
plotTitle = "Regression Line Determined by Linear Regression Model for the\n"
plotTitle += "Septal Width and Septal Length of Iris Setosa"
plotRegression(X, Y, w=model.getWeights(), title=plotTitle, \
    xLabel="Sepal Length", yLabel="Sepal Width")



# Multiple Regression example with Randomly Generated Data
print("\tMultiple Regression Example on Randomly Generated Data")
# Multiple Regression example
X, Y =make_regression(n_samples=200, n_features=2, n_targets=1, random_state=47)

# Randomly display regression data
Y = randomizeData(Y, 0.25) 

plotTitle = "Multiple Regression Demo:\n"
plotTitle += "Data Visualization for Randomly Generated Data Fitted to a Plane"
plotRegression(X, Y, title=plotTitle, xLabel="x", yLabel="y", zLabel="z")

# Create the Model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

print("\t\tSquared Error: ", end="")
print(model.squaredError(X,Y))

# Display regression plane
plotTitle = "Regression Plane Determined by Multiple Regression Model for\n"
plotTitle += "Randomly Generated Data"
plotRegression(X[:,:], Y, model.getWeights(), title=plotTitle, xLabel="x", \
     yLabel="y", zLabel="z") 



# Multiple Regression example with the Iris Plants Databse
print("\tMultiple Regression Example on the Iris Plants Database")
# Extract the first 2 features (sepal length and pedal length)
X = df.iloc[0:50, [0,2]].values
Y = df.iloc[0:50, [1]].values
plotTitle = "Multiple Regression Demo:\n"
plotTitle += "Data Visualization for the Septal Width, Septal Length, and\n"
plotTitle += "Petal Length of Iris Setosa"

# Visualize the data
plotRegression(X, Y, title=plotTitle, xLabel="Sepal Length", \
    yLabel="Petal Length", zLabel="Sepal Width")

# Create the Model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

print("\t\tSquared Error: ", end="")
print(model.squaredError(X,Y))

# Display regression plane
plotTitle = "Regression Plane Determined by Multiple Regression Model for the\n"
plotTitle += "Septal Width, Septal Length, and\n"
plotTitle += "Petal Length of Iris Setosa"
plotRegression(X, Y, w=model.getWeights(), title=plotTitle, \
    xLabel="Sepal Length", yLabel="Petal Length", zLabel="Sepal Width")
