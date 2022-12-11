import numpy as np
import pandas as pd
import sklearn
import unittest
from src import knnclassifier
class Test_knn(unittest.TestCase):
    
    def test_smoke(self):
        datasets1 = pd.read_csv('/Users/tianyue/CSE583/Project/archive/sensor_readings_4.csv')
        knnclassifier(datasets1)

    def test_oneshot1(self):
        datasets2 = pd.read_csv('/Users/tianyue/CSE583/Project/archive/sensor_readings_4.csv')
        test1 = knnclassifier(datasets2)
        ans1 = pd.read_csv('/Users/tianyue/CSE583/Project/archive/sensor_readings_4.csv',usecols=['Slight-Right-Turn'])
        def compare(list1, list2):
            error = []
            error_index = []
            if len(list1) == len(list2):
                for i in range(0, len(list1)):
                    if list1[i] == list2[i]:
                        pass
                    else:
                        error.append(abs(list1[i]-list2[i]))
                        error_index.append(i)
                        print(error)
                        print(error_index)
        compare(test1, ans1)
        
    def test_positional_indexers_are_outofbounds(self):
        with self.assertRaises(IndexError):
            datasets3 = pd.read_csv('/Users/tianyue/CSE583/Project/archive/sensor_readings_2.csv')
            knnclassifier(datasets3)
    


        

