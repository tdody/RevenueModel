#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import pandas as pd
sys.path.append('..')
import unittest
from ast import literal_eval

## import our logger module
from logger import *

## train log content
## unique_id,timestamp,tag, 
## start_date,end_date,
## x_shape,eval_test,
## model_version,model_version_note,runtime

## predict log content
## unique_id,timestamp,
## y_pred,query,
## model_version,runtime

class LoggerTest(unittest.TestCase):

    def test_01_train(self):
        """
        ensure log file is created
        """

        ## create path and delete if exists
        log_file = os.path.join(".","logs","train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## dummy info
        tag = "all"
        start_date = "2017-11-30"
        end_date = "2019-06-29"
        data_shape = (100,100)
        eval_test = {'rmse':0.5}
        run_time = "99:99:99"
        model_version = 0.1
        model_version_note = "logger test model"

        ## update log
        update_train_log(tag, start_date,
                         end_date, data_shape,
                         eval_test, run_time,
                         model_version, model_version_note,
                         test=True)

        ## test
        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        ## create path and delete if exists
        log_file = os.path.join(".","logs","train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## dummy info
        tag = "all"
        start_date = "2017-11-30"
        end_date = "2019-06-29"
        data_shape = (100,100)
        eval_test = {'rmse':0.5}
        run_time = "99:99:99"
        model_version = 0.1
        model_version_note = "logger test model"

        ## update log
        update_train_log(tag, start_date,
                         end_date, data_shape,
                         eval_test, run_time,
                         model_version, model_version_note,
                         test=True)

        ## read log
        df = pd.read_csv(log_file)

        ## read last entry
        logged_eval_test = [literal_eval(i) for i in df['eval_test'].copy()][-1]

        ## spot check
        self.assertEqual(eval_test,logged_eval_test)

    def test_03_predict(self):
        """
        ensure log file is created
        """

        ## create path and delete if exists
        log_file = os.path.join(".","logs","predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## dummy info
        y_pred=[0.5]
        query = "all_2018-01-01"
        run_time = "99:99:99"
        model_version = 0.1

        ## update log
        update_predict_log(y_pred,query,run_time,model_version,test=True)

        ## test
        self.assertTrue(os.path.exists(log_file))

    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        ## create path and delete if exists
        log_file = os.path.join(".","logs","predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## dummy info
        y_pred=[0.5]
        query = "all_2018-01-01"
        run_time = "99:99:99"
        model_version = 0.1

        ## update log
        update_predict_log(y_pred,query,run_time,model_version,test=True)

        ## read log
        df = pd.read_csv(log_file)

        ## read last entry
        logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]

        ## spot check
        self.assertEqual(y_pred,logged_y_pred)

## run test
if __name__ == '__main__':
    unittest.main()