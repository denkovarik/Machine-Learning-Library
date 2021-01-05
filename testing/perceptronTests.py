# This file runs test the Perceptron model 
#
# Usage:
#   python3 perceptronTests.py 


import io, os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from ML import Perceptron
import numpy as np
import unittest 
import math
  
class TestPerceptron(unittest.TestCase): 
  
    def test_init(self):
        """
        Tests the Perceptron class init function
        """
        # Testing Valid arguments 
        pn = Perceptron(0.1, 10)
 
        # Check that the learning rate and niter were initialized
        testRate = 0.1 
        self.assertEqual(type(pn.rate), type(testRate)) 
        self.assertEqual(pn.rate, testRate) 
        testNiter = 10
        self.assertEqual(type(pn.niter), type(testNiter)) 
        self.assertEqual(pn.niter, 10)

        # Check that the errors and weight array was initialized to something 
        testArray = np.array([0])
        self.assertEqual(len(pn.errors), 1)
        self.assertEqual(type(pn.errors), type(testArray)) 
        self.assertEqual(len(pn.weight),2)
        self.assertEqual(type(pn.weight), type(testArray))
 
        # Testing invalid argument for learning rate
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        pn = Perceptron(1, 10)
        sys.stdout = sys.__stdout__ # now restore stdout function
        
        self.assertEqual(pn.rate, 0.1) 
        self.assertEqual(pn.niter, 10)
 
        # Testing invalid argument for niter
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        pn = Perceptron(0.1, 0.10)
        sys.stdout = sys.__stdout__ # now restore stdout function
        self.assertEqual(pn.niter, 10) 
        self.assertEqual(pn.rate, 0.1)


    def test_dotProduct(self):
        """
        Tests the Perceptron class function dotProduct() that computes the dot
        product for the attributes of x and the perceptron weight
        """
        pn = Perceptron(0.1, 10)
        x = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        y = np.array([-1, -1, -1, -1])
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        self.assertEqual(math.isnan(pn.dotProduct(x)), math.isnan(math.nan))
        sys.stdout = sys.__stdout__ # now restore stdout function

        # Testing with valid sizes
        pn.weight = np.array([-0.31,0.7,0.1])
        self.assertEqual(abs(pn.dotProduct(x[0]) - 0.1) < 0.001, True)
        self.assertEqual(abs(pn.dotProduct(x[1]) - 0.49) < 0.001, True)
        self.assertEqual(abs(pn.dotProduct(x[2]) - 0.8799999999999999) < 0.001, True)
        self.assertEqual(abs(pn.dotProduct(x[3]) - 1.2699999999999998) < 0.001, True)
 

    def test_fit(self):
        """
        Tests the Perceptron class function fit() function.
        """
        pn = Perceptron(0.1, 10)

        # Testing x parm not a numpy array
        x = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        y = np.array([-1, -1, -1, -1])
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        result = pn.fit(x, y)
        sys.stdout = sys.__stdout__ # now restore stdout function
        self.assertEqual(result[0], "x must be of type numpy array")

        # Testing x doesn't have the right dimensions
        x = np.array([-1, -1, -1, -1])
        y = np.array([-1, -1, -1, -1])
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        result = pn.fit(x, y)
        sys.stdout = sys.__stdout__ # now restore stdout function
        self.assertEqual(result[0], "x must have 2 dimensions")
        
        # Testing y parm not a numpy array
        y = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        x = np.array([-1, -1, -1, -1])
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        result = pn.fit(x, y)
        sys.stdout = sys.__stdout__ # now restore stdout function
        self.assertEqual(result[0], "y must be of type numpy array")

        # Testing y doesn't have the right dimensions
        y = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        result = pn.fit(x, y)
        sys.stdout = sys.__stdout__ # now restore stdout function
        self.assertEqual(result[0], "y must have 1 dimension")

        # Testing that the sizes of x and y don't match
        x = np.array([[5.1, 1.4], [4.9, 1.4], [4.7, 1.3], [4.6, 1.5]])
        y = np.array([-1, -1, -1, -1, -1])
        text_trap = io.StringIO() # create a text trap and redirect stdout
        sys.stdout = text_trap
        result = pn.fit(x, y)
        sys.stdout = sys.__stdout__ # now restore stdout function
        #self.assertEqual(result[0], "sizes of x and y must match")

        # Testing valid parameters
        pn = Perceptron(0.1, 10)
        x = np.array([[5.1, 1.4], [4.9, 1.4], [4.7, 1.3], [4.6, 1.5], \
            [5.7, 4.2], [6.2, 4.3], [5.1, 3.0], [5.7, 4.1]])
        y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        # Check the initial set of the weight array
        self.assertEqual(np.array_equal(pn.weight, np.array([0,0])), True)
        # Fit the weights to the data
        pn.fit(x, y)
        # Check that the size of the weight array was updated based on size of x and y
        self.assertEqual(len(y), len(x))
        self.assertEqual(len(pn.weight), len(x[0]) + 1)


    def test_net_input(self):
        pn = Perceptron(0.1, 10)
        x = np.array([[5.1, 1.4], [4.9, 1.4], [4.7, 1.3], [4.6, 1.5], \
            [5.7, 4.2], [6.2, 4.3], [5.1, 3.0], [5.7, 4.1]])
        y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        # Fit the weights to the data
        pn.fit(x, y)

    
    def test_predict(self):
        """
        Tests the Perceptron class function predict() function.
        """
        pn = Perceptron(0.1, 10)
        x = np.array([[5.1, 1.4], [4.9, 1.4], [4.7, 1.3], [4.6, 1.5], \
            [5.7, 4.2], [6.2, 4.3], [5.1, 3.0], [5.7, 4.1]])
        y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        
        # Test calling the predict function
        pn.fit(x, y)
        predictions = pn.predict(x)
        self.assertEqual(np.array_equal(predictions, y), True)


if __name__ == '__main__': 
    unittest.main() 

