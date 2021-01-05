# File: logisticRegressionDemo.py
# Author: Dennis Kovarik
# Purpose: Run the Logistic Regression model on examples
# Usage: python3 logisticRegressionDemo.py

# Import libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from ML import LogisticRegression 
from utils import plotLogisticRegressionDecisionRegions



print("Running Logistic Regression Demo")

# Download and read the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Extract the first 100 class labels
y = df.iloc[0:100, 4].values

# Convert labels to Y = {0, 1}
y = np.where(y == 'Iris-setosa', 0, 1)

# Extract the features (sepal length and pedal length)
X = df.iloc[0:100, [0 , 2]].values

output = "\tUsing Logistic Regression Model to Perform Binary Classification "
output += "on Iris Setosa "
output += "and Iris Versicolor Based on Septal Length and Petal Length"
print(output)

xLabel = "sepal length"
yLabel = "petal length"
class1 = "Iris-setosa"
class2 = "Iris-versicolor"

# Visualize the data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label=class1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label=class2)
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.legend(loc='upper left')
title = "Logistic Regression Demo:\n"
title += "Visualization of Petal Length and Septal Length\n" 
title += "for Iris Setosa and Iris Versicolor"
plt.title(title)
plt.show()

# Find the best fit line to seperate the classes
model = LogisticRegression(alpha=0.1, epochs=1000)

# calls the logistic regression method
weight, cost_history_list = model.fit(X, y, 4)

# visualize the training loss
plt.plot(np.arange(len(cost_history_list)), cost_history_list)
plt.xlabel("Epochs")
plt.ylabel("Cost")
title = "Logistic Regression Model Training Loss using\n"
title += "Stochastic Gradient Descent"
plt.title(title)
plt.show()

# Plot the decision regions determined by the model
title = "Decision Boundary Determined by the Logistic Regression Model for\n"
title += "the Classificiation of Iris Setosa and Iris Versicolor Based on\n"
title += "Sepal Length and Petal Length"
plotLogisticRegressionDecisionRegions(X, y, model, title=title, \
    xLabel=xLabel, yLabel=yLabel, class1=class1, \
    class2=class2) 



# Trying Logistic Regression model on same labels but using the sepal width and 
# petal length features
output = "\tUsing Logistic Regression Model to Perform Binary Classification "
output += "on Iris Setosa and Iris Versicolor Based on Septal Width and "
output += "Petal Width"
print(output)

# Extract the first 100 class labels
y = df.iloc[0:100, 4].values

# Convert labels to Y = {0, 1}
y = np.where(y == 'Iris-setosa', 0, 1)

# Extract the first 2 features (sepal length and pedal length)
X = df.iloc[0:100, [1, 3]].values

xLabel = "sepal width"
yLabel = "petal width"
class1 = "Iris-setosa"
class2 = "Iris-versicolor"

# Visualize the data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label=class1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label=class2)
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.legend(loc='upper left')
title = "Logistic Regression Demo:\n"
title += "Visualization of Petal Width and Septal Width\n" 
title += "for Iris Setosa and Iris Versicolor"
plt.title(title)
plt.show()

# Find the best fit line to seperate the classes
model = LogisticRegression(alpha=0.1, epochs=1000)

# calls the logistic regression method
weight, cost_history_list = model.fit(X, y, 4)

# visualize the training loss
plt.plot(np.arange(len(cost_history_list)), cost_history_list)
plt.xlabel("Epochs")
plt.ylabel("Cost")
title = "Logistic Regression Model Training Loss using\n"
title += "Stochastic Gradient Descent"
plt.title(title)
plt.show()

# Plot the decision regions determined by the model
title = "Decision Boundary Determined by Logistic Regression Model for the\n"
title += "Classificiation of Iris Setosa and Iris Versicolor Based on Sepal\n"
title += "Width and Petal Width"
plotLogisticRegressionDecisionRegions(X, y, model, title=title, \
    xLabel=xLabel, yLabel=yLabel, class1=class1, \
    class2=class2) 



# Trying Logistic Regression model on same labels but using the sepal width and 
# petal length features
output = "\tUsing Logistic Regression Model to Perform Binary Classification "
output += "on Iris Versicolor and Iris Verginica Based on Septal Length and "
output += "Petal Length"
print(output)

# Extract the labels from 51 through 150
y = df.iloc[51:150, 4].values

# Convert labels to Y = {0, 1}
y = np.where(y == 'Iris-versicolor', 0, 1)

# Extract the first 2 features (sepal length and pedal length)
X = df.iloc[51:150, [0 , 2]].values

xLabel = "sepal length"
yLabel = "petal length"
class1 = "Iris-versicolor"
class2 = "Iris-virginica"

# Visualize the data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label=class1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label=class2)
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.legend(loc='upper left')
title = "Logistic Regression Demo:\n"
title += "Visualization of Petal Length and Septal Length\n" 
title += "for Iris Virginica and Iris Versicolor"
plt.title(title)
plt.show()

# Find the best fit line to seperate the classes
model = LogisticRegression(alpha=0.2, epochs=1000)

# calls the logistic regression method
weight, cost_history_list = model.fit(X, y, 10)

# visualize the training loss
plt.plot(np.arange(len(cost_history_list)), cost_history_list)
plt.xlabel("Epochs")
plt.ylabel("Cost")
title = "Logistic Regression Model Training Loss using\n"
title += "Stochastic Gradient Descent"
plt.title(title)
plt.show()

# Plot the decision regions determined by the model
title = "Decision Boundary Determined by the Logistic Regression Model for\n"
title += "the Classificiation of Iris Virginica and Iris Versicolor Based on\n"
title += "Sepal Length and Petal Length"
plotLogisticRegressionDecisionRegions(X, y, model, title=title, \
    xLabel=xLabel, yLabel=yLabel, class1=class1, \
    class2=class2) 
