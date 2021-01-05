# File: axisAlignedRectangleEx.py
# Author: Dennis Kovarik
# Purpose: Run the Axis Aligned Rectangles model on examples
# Usage: python3 axisAlignedRectangleEx.py


import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from ML import AxisAlignedRectangles
from generatedData import *


print("Running Axis Aligned Rectangles Demo")

# Axis Aligned Rectangle Demo in 2D
print("\tClassification for Generated Data using 2 Attributes")
x = axisAlignedRectangleEx2D.copy()
y = axisAlignedRectangleEx2DLabels.copy()

# Visualize the data
plotTitle = "Axis Aligned Rectangles Demo:\n"
plotTitle += "Visualization of Generated Data"
xLabel = "X"
yLabel = "Y"
plotAxisAlignedRectangles2D(x, y, title=plotTitle, xLabel=xLabel, yLabel=yLabel)

model = AxisAlignedRectangles()

model.fit(x, y)

predictions = model.predict(x)
print("\t\tNumber of Misclassified Points: ", end="")
print(len(np.where(predictions != y)[0]))

# Visualize the data
plotTitle = "Classification of Generated Data using the\n"
plotTitle += "Axis Aligned Rectangles Model"
plotAxisAlignedRectangles2D(x, y, model.getWeights(), title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel)



# Axis Aligned Rectangle Demo in 2D
print("\tClassification for Iris Versicolor using Sepal Length and Petal Length")
# Download and read the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Extract the first 100 class labels
y = df.iloc[0:150, 4].values

# Convert labels to Y = {-1, 1}
y = np.where(y == 'Iris-versicolor', 1, -1)

# Extract the first 2 features (sepal length and pedal length)
x = df.iloc[0:150, [0 , 2]].values

plotTitle = "Axis Aligned Rectangles Demo:\n"
plotTitle += "Visualization of Sepal Length and Petal Length for Plants from\n"
plotTitle += "the Iris Dataset"
xLabel = "Sepal Length"
yLabel = "Petal Length"

# Visualize the data
plotAxisAlignedRectangles2D(x, y, title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel, \
    positiveClassLabel="Iris-versicolor", negativeClassLabel="Other")

# Create and fit model to data
model = AxisAlignedRectangles()
model.fit(x, y)
predictions = model.predict(x)
print("\t\tNumber of Misclassified Points: ", end="")
print(len(np.where(predictions != y)[0]))

# Visualize the data
plotTitle = "Classification of Iris Versicolor based on Sepal Length and\n"
plotTitle += "Petal Length using the Axis Aligned Rectangles Model"
plotAxisAlignedRectangles2D(x, y, model.getWeights(), title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel, positiveClassLabel="Iris-versicolor", \
    negativeClassLabel="Other")



# Axis Aligned Rectangle Demo in 3D
print("\tClassification for Generated Data using 3 Attributes")
x = axisAlignedRectangleEx3D.copy()
y = axisAlignedRectangleEx3DLabels.copy()

plotTitle = "Axis Aligned Rectangles Demo:\n"
plotTitle += "Visualization of Generated Data"
xLabel = "X"
yLabel = "Y"
zLabel = "Z"

# Visualize the data
plotAxisAlignedRectangles3D(x, y, title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel)

model = AxisAlignedRectangles()

model.fit(x, y)

predictions = model.predict(x)
print("\t\tNumber of Misclassified Points: ", end="")
print(len(np.where(predictions != y)[0]))

# Visualize the data
plotTitle = "Classification of Generated Data using the\n"
plotTitle += "Axis Aligned Rectangles Model"
plotAxisAlignedRectangles3D(x, y, model.getWeights(), title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel)



# Axis Aligned Rectangle Demo in 3D
print("\tClassification for Iris Versicolor using Sepal Length, Sepal Width, and Petal Length")

# Extract the first 100 class labels
y = df.iloc[0:150, 4].values

# Convert labels to Y = {-1, 1}
y = np.where(y == 'Iris-versicolor', 1, -1)

# Extract the first 2 features (sepal length and pedal length)
x = df.iloc[0:150, [0, 1, 2]].values

plotTitle = "Axis Aligned Rectangles Demo:\n"
plotTitle += "Visualization of Sepal Length, Sepal Width and Petal Length for\n"
plotTitle += "Plants from the Iris Dataset"
xLabel = "Sepal Length"
yLabel = "Sepal Width"
zLabel = "Petal Length"

# Visualize the data
plotAxisAlignedRectangles3D(x, y, title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel, zLabel=zLabel, \
    positiveClassLabel="Iris-versicolor", negativeClassLabel="Other")

# Create and fit model to data
model = AxisAlignedRectangles()
model.fit(x, y)
predictions = model.predict(x)
print("\t\tNumber of Misclassified Points: ", end="")
print(len(np.where(predictions != y)[0]))

# Visualize the data
plotTitle = "Classification of Iris Versicolor based on Sepal Length, Petal\n"
plotTitle += "Width, and Petal Length using the Axis Aligned Rectangles Model"
plotAxisAlignedRectangles3D(x, y, model.getWeights(), title=plotTitle, \
    xLabel=xLabel, yLabel=yLabel, zLabel=zLabel, \
    positiveClassLabel="Iris-versicolor", negativeClassLabel="Other")
