# Machine Learning Library

<p align="center">
  <img src="https://github.com/denkovarik/Machine-Learning-Library/blob/master/images/Logistic_Regression_Decision_Boundary_Ex1.png" width="400" title="Logistic Regression Decision Boundary">
  <img src="https://github.com/denkovarik/Machine-Learning-Library/blob/master/images/Linear_Regression_Regression_Ex4.png" width="400" title="Multiple Regression Fitted Plane">
  <img src="https://github.com/denkovarik/Machine-Learning-Library/blob/master/images/KNN_Decision_Boundary_k%3D5.png" width="400" title="K Nearest Neighbors Decision Boundaries">
</p>

## Introduction
This project consists of a collection of machine learning modules written in Python. My name is Dennis Kovarik, and one of my interests is in machine learning. Machine learning is the development and application of algorithms that allow computers to find patterns in large datasets to make predictions based on them. Machine learning has many applications accross multiple industries, and it can allow companies and organizations to provide better products and services by  leveraging the huge amounts of data available to them. To learn more about this subject, I developed this library of machine learning modules. This collection consists of the following modules: 

* Perceptron
* Linear Regression
* Logistic Regression
* K-Nearest Neighbor
* Axis Aligned Rectangles  

All demonstrations included in this repository runs the above modules on subsets of the Iris Dataset.

## Setup
This library was developed and tested on the Linux Command line in Ubuntu 20.04
### Dependencies
* Python 3
  * math
  * random
  * mpl_toolkits.mplot3d
  * io
  * os
  * sys
  * inspect
  * unittest
* NumPy
  * numpy.linalg
* Matplotlib
  * matplotlib.pyplot
  * matplotlib.colors
* seaborn
* Pandas
* scikit-learn
  * sklearn.datasets
* Bash

### Optional Software for Windows Users
* Xming
   * X-server package for displaying platform on windows
   
### Clone the Repo
* SSH
```
git clone git@github.com:denkovarik/Machine-Learning-Library.git
```
* HTTPS
```
git clone https://github.com/denkovarik/Machine-Learning-Library.git
```

## Testing
Testing was only completed for the Perceptron and K Nearest Neighbors modules. You can run the tests by running the following command.
```
./testing/runTests.sh
```

## Usage
### Run All Demos
```
./runAllDemos.sh
```
### Run the Perceptron Module Demo
```
python3 perceptronDemo.py
```
### Run the Linear Regression Module Demo
```
python3 linearRegressionDemo.py
```
### Run the Logistic Regression Module Demo
```
python3 logisticRegressionDemo.py
```
### Run the K Nearest Neighbors Module Demo
```
python kNearestNeighborsDemo.py
```
### Run the Axis Aligned Rectangles Module Demo
```
python3 axisAlignedRectangleEx.py
```

## Author
* Dennis Kovarik 

## References
(2019). UNDERSTANDING MACHINE LEARNING FROM THEORY TO ALGORITHMS [Review of UNDERSTANDING MACHINE LEARNING FROM THEORY TO ALGORITHMS]. CAMBRIDGE UNIVERSITY PRESS. (Original work published 2014)

Multiple Linear Regression using OLS and gradient descent. (2019, July 22). AI ASPIRANT. https://aiaspirant.com/multiple-linear-regression/

Kalu, D. P. C. (2020, October 14). How to Implement Logistic Regression with Python. Neuraspike. https://neuraspike.com/blog/logistic-regression-python-tutorial/

Brownlee, J. (2016, October 30). How To Implement Logistic Regression From Scratch in Python. Machine Learning Mastery. https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/

Brownlee, J.. (2018, April 14). Tutorial To Implement k-Nearest Neighbors in Python From Scratch. Machine Learning Mastery. https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

