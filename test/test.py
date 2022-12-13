"""
python -m unittest test/svm_test.py
"""
import numpy as np
import pandas as pd
import sklearn
import unittest
from src import SVM_learning
from src import knnclassifier
from src import NNetclassifier
from src import Linclassifier



class test(unittest.TestCase):
    
    def test_smoke(self):
        datasets1 = pd.read_csv('test/test_4.csv')
        SVM_learning(datasets1)

    def test_oneshot1(self):
        datasets2 = pd.read_csv('test/test_4.csv')
        test1 = SVM_learning(datasets2)
        ans1 = pd.read_csv('test/test_4.csv',usecols=['Sharp-Right-Turn'])
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
        with self.assertRaises(ValueError):
            datasets3 = pd.read_csv('test/test_2.csv')
            SVM_learning(datasets3)
    

    def test_smoke2(self):
        datasets1 = pd.read_csv('test/test_4.csv')
        knnclassifier(datasets1)

    def test_oneshot2(self):
        datasets2 = pd.read_csv('test/test_4.csv')
        test1 = knnclassifier(datasets2)
        ans1 = pd.read_csv('test/test_4.csv',usecols=['Sharp-Right-Turn'])
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
        
    def test_positional_indexers_are_outofbounds2(self):
        with self.assertRaises(IndexError):
            datasets3 = pd.read_csv('test/test_2.csv')
            knnclassifier(datasets3)

    
    def test_smoke_nnet(self):
        datasets1 = pd.read_csv('test/test_4.csv')
        NNetclassifier(datasets1)

    def test_oneshot1_nnet(self):
        datasets2 = pd.read_csv('test/test_4.csv')
        test1 = NNetclassifier(datasets2)
        ans1 = pd.read_csv('test/test_4.csv',usecols=['Sharp-Right-Turn'])
        compare(test1, ans1)
        
    def test_positional_indexers_are_outofbounds_nnet(self):
        with self.assertRaises(IndexError):
            datasets3 = pd.read_csv('test/test_2.csv')
            NNetclassifier(datasets3)
    
    def test_smoke_lin(self):
        datasets1 = pd.read_csv('test/test_4.csv')
        Linclassifier(datasets1)

    def test_oneshot1_lin(self):
        datasets2 = pd.read_csv('test/test_4.csv')
        test1 = Linclassifier(datasets2)
        ans1 = pd.read_csv('test/test_4.csv',usecols=['Sharp-Right-Turn'])
        compare(test1, ans1)
        
    def test_positional_indexers_are_outofbounds_lin(self):
        with self.assertRaises(IndexError):
            datasets3 = pd.read_csv('test/test_2.csv')
            Linclassifier(datasets3)

