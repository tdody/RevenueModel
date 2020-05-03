#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time
import os
import re
import csv
import sys
import uuid
import joblib
from datetime import date, datetime
import pandas as pd

## create path for logs if needed
if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(tag, start_date, end_date, data_shape, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):

    """
    update training log
    """

    ## define cyclic name for log
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))

    ## write the data to csv file
    header = ["unique_id", "timestamp", "tag", "start_date", "end_date", "x_shape", "eval_test", "model_version",
              "model_version_note", "runtime"]
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)
        to_write = map(str,[
            uuid.uuid4(),time.time(),tag,start_date,end_date,data_shape,eval_test,
            MODEL_VERSION,MODEL_VERSION_NOTE,runtime])
        writer.writerow(to_write)

def update_predict_log(y_pred,query,runtime,MODEL_VERSION,test=False):
    """
    update predict log file
    """

    ## define cyclic name for log
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','query','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),y_pred,query,
                            MODEL_VERSION,runtime])
        writer.writerow(to_write)

def find_latest_predict_log(train=True):
    """
    Fetch latest log and return it into a formatted pandas dataframe
    """

    ## main log locations    
    log_dir = './logs'
    
    ## set prefix train or predict
    if train:
        prefix = "train-test"
    else:
        prefix = "predict"

    ## find all relevant logs
    logs = [f for f in os.listdir(log_dir) if re.search(prefix,f)]

    if len(logs)==0:
        return None
    else:
        ## find most recent
        logs.sort()
        df = pd.read_csv(os.path.join(log_dir,logs[-1]))

        if train:
            ## filter columns
            columns = ['timestamp', 'tag','start_date', 'end_date', 'x_shape', 'eval_test', 'model_version', 'runtime']

        else:
            ## filter columns
            columns = ['timestamp', 'y_pred', 'query', 'model_version', 'runtime']

        ## format time stamp
        df['timestamp'] = pd.Series([datetime.fromtimestamp(x) for x in df['timestamp']]).dt.strftime('%m-%d-%Y %r')

        return df[columns]


if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE
    
    ## train logger
    update_train_log("test","01/01/1990","12/31/1990",str((100,10)),"{'rmse':0.5}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    ## predict logger
    update_predict_log("[0]","[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
                       "00:00:01",MODEL_VERSION, test=True)