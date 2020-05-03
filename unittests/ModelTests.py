#!/usr/bin/env python
"""
model tests
"""
import sys, os
sys.path.append('..')
import unittest

## import our model module
from model import *

MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

class ModelTest(unittest.TestCase):
    """
    Test the model functionalities (train , load, and predict)
    """

    def test_01_train(self):
        """
        Test train functionality
        """
        ## locate data
        data_dir = os.path.join(".","data","cs-train")
    
        ## train model
        model_train(data_dir,test=True)

        ## check if model has been saved
        self.assertTrue(os.path.isfile("./models/test-all-0_1.joblib"))

    def test_02_load(self):
        """
        Test load functionality
        """
        ## load model
        all_data, all_models = model_load(prefix='test', data_dir=os.path.join(".","data","cs-train"))

        self.assertTrue('predict' in dir(all_models['all']))
        self.assertTrue('fit' in dir(all_models['all']))

    def test_03_predict(self):
        """
        Test predict functionality
        """
        import numpy as np

        ## load model
        all_data, all_models = model_load(prefix='test', data_dir=os.path.join(".","data","cs-train"))

        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day,all_models=all_models,all_data=all_data,test=True)

        y_pred = result['y_pred']
        self.assertTrue(type(y_pred[0]) is np.float64)

## run test
if __name__ == '__main__':
    unittest.main()