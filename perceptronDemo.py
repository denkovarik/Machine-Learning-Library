# File: perceptronDemo.py 
# Author: Dennis Kovarik
# Purpose: Run the Perceptron model on examples
# Usage: python3 perceptronDemo.py 

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ML import Perceptron
from utils import plotPerceptronDecisionRegions

print("Running Perceptron Demo")
# Download and read the data
dataUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'
dataUrl += 'iris.data'
df = pd.read_csv(dataUrl, header=None)

# Using perceptron to perform binary classification on Iris Setosa and Iris 
# Versicolor based on their septal length and petal length 
output = "\tUsing Perceptron to Perform Binary Classification on Iris Setosa "
output += "and Iris Versicolor Based on Septal Length and Petal Length"
print(output)

# Extract the first 100 class labels
y = df.iloc[0:100, 4].values

# Convert labels to Y = {-1, 1}
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract the first 2 features (sepal length and pedal length)
x = df.iloc[0:100, [0 , 2]].values

# Visualize the data
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Iris-versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
title = "Perceptron Demo:\n"
title += "Data Visualization for the Sepal Length and Petal Length of\nIris "
title += "Setosa and Iris Versicolor"
plt.title(title)
plt.show()

# Create Perceptron and fit it to the data
pn = Perceptron(0.1, 10)
pn.fit(x,y)   

# Plot the perceptron errors per iteration
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.title("Errors per Iteration for Perceptron Training")
plt.xlabel('Iteration')
plt.ylabel('Number of misclassifications')
plt.show()

# Plot the decision boundary
title = "Decision Boundary Determined by Perceptron Model for Classifying\n"
title += "Iris Setosa and Iris Versicolor Based on Sepal Length and Petal Length"
xLabel = "sepal length"
yLabel = "petal length"
plotPerceptronDecisionRegions(x, y, pn, title, xLabel, yLabel, \
     class1="Iris-setosa", class2="Iris-versicolor") 



# Trying perceptron on same labels but using the sepal width and petal length features
output = "\tUsing Perceptron to Perform Binary Classification on Iris Setosa "
output += "and Iris Versicolor Based on Septal Width and Petal Width"
print(output)

# Extract the first 100 class labels
y = df.iloc[0:100, 4].values

# Convert labels to Y = {-1, 1}
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract the first 2 features (sepal length and pedal length)
x = df.iloc[0:100, [1, 3]].values

# Visualize the data
xLabel = "sepal width"
yLabel = "petal width"
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Iris-versicolor')
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.legend(loc='upper left')
title = "Perceptron Demo:\nData Visualization for the Sepal Width and Petal "
title += "Width of\nIris Setosa and Iris Versicolor"
plt.title(title)
plt.show()

# Create Perceptron and fit it to the data
pn = Perceptron(0.1, 10)
pn.fit(x,y)   

# Plot the perceptron errors per iteration
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.title("Errors per Iteration for Perceptron Training")
plt.xlabel('Iteration')
plt.ylabel('Number of Misclassifications')
plt.show()

title = "Decision Boundary Determined by Perceptron Model for Classifying\n"
title += "Iris Setosa and Iris Versicolor Based on Sepal Width and Petal Width"
plotPerceptronDecisionRegions(x, y, pn, title=title, xLabel=xLabel, \
     yLabel=yLabel, class1="Iris-setosa", class2="Iris-versicolor") 



output = "\tUsing Perceptron to Perform Binary Classification on Iris Versicolor"
output += " and Iris Virginica Based on Septal Length and Petal Length"
print(output)

# Extract the labels from 51 through 150
y = df.iloc[51:150, 4].values

# Convert labels to Y = {-1, 1}
y = np.where(y == 'Iris-versicolor', -1, 1)

# Extract the first 2 features (sepal length and pedal length)
x = df.iloc[51:150, [0 , 2]].values

xLabel = "sepal length"
yLabel = "petal length"

# Visualize the data
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Iris-versicolor')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Iris-virginica')
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.legend(loc='upper left')
title = "Perceptron Demo:\nData Visualization for the Sepal Length and Petal "
title += "Length of\nIris Versicolor and Iris Virginica"
plt.title(title)
plt.show()

# Create Perceptron and fit it to the data
pn = Perceptron(0.1, 1000)
pn.fit(x,y)   

# Plot the perceptron errors per iteration
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.title("Errors per Iteration for Perceptron Training")
plt.xlabel('Iteration')
plt.ylabel('Number of Misclassifications')
plt.show()

title = "Decision Boundary Determined by Perceptron Model for Classifying\n"
title += "Iris Virginica and Iris Versicolor Based on Sepal Length and\n"
title += "Petal Length"
plotPerceptronDecisionRegions(x, y, pn, title=title, xLabel=xLabel, \
 yLabel=yLabel, class1="Iris-versicolor", class2="Iris-virginica") 
