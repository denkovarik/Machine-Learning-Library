# This file runs test the K Nearest Neighbors model 
#
# Usage:
#   python3 testKNN.py 


import io,os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import unittest
from ML import KNearestNeighbors
import numpy as np

  
class TestKNN(unittest.TestCase):

    def test_distance(self):
        """
        This function tests the KNN's distance function. 
        """
        x = np.zeros(10)
        y = np.zeros(10)
        model = KNearestNeighbors(1)
        # Calculating distance for object with one feature
        row1 = [1]
        row2 = [1]
        self.assertTrue(np.isclose(model.distance(row1, row2), 0.0, 0.001))
        
        row1 = [1]
        row2 = [2]
        self.assertTrue(np.isclose(model.distance(row1, row2), 1.0, 0.001))
        
        row1 = [2]
        row2 = [1]
        self.assertTrue(np.isclose(model.distance(row1, row2), 1.0, 0.001))
        
        # Calculating distance for object with two features
        row1 = [1,1]
        row2 = [1,1]
        self.assertTrue(np.isclose(model.distance(row1, row2), 0.0, 0.001))
        row1 = [1,2]
        row2 = [2,6]
        self.assertTrue(np.isclose(model.distance(row1, row2), 4.1231, 0.001))
        
        # Calculating distance for object with three features
        row1 = [1,1,1]
        row2 = [1,1,1]
        self.assertTrue(np.isclose(model.distance(row1, row2), 0.0, 0.001))
        row1 = [7,4,3]
        row2 = [17,6,2]
        self.assertTrue(np.isclose(model.distance(row1, row2), 10.2469, 0.001))
        row1 = [7.4,4.1,3.2]
        row2 = [17.7,6.3,2.5]
        self.assertTrue(np.isclose(model.distance(row1, row2), 10.555567, 0.001))
        
    

if __name__ == '__main__':
    unittest.main() 
