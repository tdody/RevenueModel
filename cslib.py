#!/usr/bin/env python

"""
Module to read and import data
"""
import os
import re
import time
import numpy as np
import pandas as pd
import shutil
from collections import defaultdict

def fetch_data(data_dir):
    """
    Collect the data stored into json files.
    Return a pandas DataFrame containing the combined json files.
    """

    ## check if data_dire is valid
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist")
    if len(os.listdir(data_dir)) == 0:
        raise Exception("specified data dire does not contain any file")

    ## collect all files contained in data_dir
    dir_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search("\.json", f)]
    
    ## store the content of each file
    all_months = {}
    for f in dir_files:
        df = pd.read_json(f)
        all_months[os.path.split(f)[-1]] = df
    
    ## column names to be used
    clean_columns = ["country","customer_id","invoice","price",
                     "stream_id","times_viewed","year","month","day"]
    clean_columns = sorted(clean_columns)
    
    ## clean column names
    ## StreamID -> stream_id
    ## TimesViewed -> times_viewed
    ## total_price -> price
    for file, df in all_months.items():
        cols = df.columns.tolist()
        if "StreamID" in cols:
            df.rename(columns={"StreamID": "stream_id"}, inplace=True)
        if "TimesViewed" in cols:
            df.rename(columns={"TimesViewed": "times_viewed"}, inplace=True)
        if "total_price" in cols:
            df.rename(columns={"total_price": "price"}, inplace=True)
    
        ## verify column compatibility
        if sorted(df.columns.tolist()) != clean_columns:
            raise Exception("column names do not match schema")
    
    ## merge dataframes
    df = pd.concat(list(all_months.values()))
    
    ## clean invoice
    df["invoice"] = [re.sub("\D+","",f) for f in df["invoice"].values]

    ## eliminate outliers
    df = df[(df["price"]>=0) & (df["price"]<=1000)]
    
    ## create date
    df["invoice_date"] = df["year"].astype("str") + "/" + df["month"].astype("str").str.zfill(2) + "/" + df["day"].astype("str").str.zfill(2)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    
    ## sort by date and reset the index
    df.sort_values(by='invoice_date',inplace=True)
    df.reset_index(drop=True,inplace=True)

    ## return structured dataframe containing all transactions (no grouping)
    return df

def convert_to_ts(df_orig, country=None):
    """
    Takes original dataframe as input.
    Return aggregated time series over each day
    """
    
    ## filter for country
    if country:
        if country not in np.unique(df_orig["country"].values):
            raise Exception("country not found")
            
        mask = df_orig["country"] == country
        df = df_orig[mask]
    else:
        df = df_orig
        
    ## sample daily data and ensure all days are accounted for
    start_date = df["invoice_date"].min()
    end_date = df["invoice_date"].max()
    days = np.arange(start_date,end_date,dtype='datetime64[D]')

    ## sample the data for each day
    purchases = np.array([np.where(df["invoice_date"]==day)[0].size for day in days])
    invoices = [np.unique(df[df["invoice_date"]==day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df["invoice_date"]==day]['stream_id'].values).size for day in days]
    views =  [df[df["invoice_date"]==day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df["invoice_date"]==day]['price'].values.sum() for day in days]
    year_month = ["-".join(re.split("-",str(day))[:2]) for day in days]

    df_time = pd.DataFrame({'date':days,
                            'purchases':purchases,
                            'unique_invoices':invoices,
                            'unique_streams':streams,
                            'total_views':views,
                            'year_month':year_month,
                            'revenue':revenue})

    return df_time

def fetch_ts(data_dir, clean=False, country=None):
    """
    convenience function to read in new data
    uses csv to load quickly
    use clean=True when you want to re-create the files
    """
    ## set data location
    ts_data_dir = os.path.join(data_dir,"ts-data")
    
    ## delete folder if clean
    if clean:
        if os.path.exists(ts_data_dir) and os.path.isdir(ts_data_dir):
            shutil.rmtree(ts_data_dir)

    ## make folder if does not exist
    if not os.path.exists(ts_data_dir):
        os.mkdir(ts_data_dir)
        
    ## if files have already been processed load them  
    if country is None:
        if len(os.listdir(ts_data_dir)) > 0:
            ## crete dictionary {country:dataframe)
            print("... loading ts data from files")
            return({re.sub("\.csv","",cf)[3:]:pd.read_csv(os.path.join(ts_data_dir,cf), parse_dates=['date']) for cf in os.listdir(ts_data_dir) if cf!="ts-log.csv"})
    else:
        ## ts-netherlands.csv
        country_id = re.sub("\s+","_",country.lower())
        print(os.path.join(ts_data_dir, "ts-"+country_id+".csv"))
        if os.path.isfile(os.path.join(ts_data_dir, "ts-"+country_id+".csv")):
            print("... loading ts data for '{}'".format(country))
            return {country_id: pd.read_csv(os.path.join(ts_data_dir, "ts-"+country_id+".csv"), parse_dates=['date'])} 

    ## get original data as one full dataframe
    print("...processing data for loading")
    df = fetch_data(data_dir)

    ## find the top 10 countries (wrt revenue)
    table = pd.pivot_table(df, values="price", index="country", aggfunc="sum")
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue',inplace=True,ascending=False)
    top_ten_countries = np.array(list(table.index))[:10]
    
    ## load the data for all countries and top 10
    dfs = {}
    dfs["all"] = convert_to_ts(df)
    for country in top_ten_countries:
        ## format country name (remove spaces)
        country_id = re.sub("\s+","_",country.lower())
        ## store dataframe
        dfs[country_id] = convert_to_ts(df,country=country)

    ## save country names
    countries = top_ten_countries.tolist()
    countries.append("ALL")
    pd.Series(np.array(countries)).sort_values(ascending=True).reset_index(drop=True).to_csv(os.path.join(ts_data_dir, "ts-log.csv"),header=None)

    ## save the data as csvs
    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir,"ts-"+key+".csv"),index=False)
            
    return dfs

def engineer_features(df,training=True):
    """
    for any given day the target becomes the sum of the next days revenue
    for that day we engineer several features that help predict the summed revenue
    
    the 'training' flag will trim data that should not be used for training
    when set to false all data will be returned
    """

    ## extract dates
    dates = df['date'].values.copy()
    dates = dates.astype('datetime64[D]')

    ## engineer some features
    eng_features = defaultdict(list)
    previous =[7, 14, 21, 28, 35, 42, 49, 56, 63, 70] # [7, 14, 28, 70]
    y = np.zeros(dates.size)
    for d,day in enumerate(dates):

        ## use windows in time back from a specific date
        for num in previous:
            ## get current day
            current = np.datetime64(day, 'D') 
            ## get first day of N previous days
            prev = current - np.timedelta64(num, 'D')
            ## filer days to only keep desired ones
            mask = np.in1d(dates, np.arange(prev,current,dtype='datetime64[D]'))
            ## sum revenue over selected period
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        ## get get the target revenue
        ## target = revenue over next 30 days
        plus_30 = current + np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(current,plus_30,dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        ## attempt to capture monthly trend with previous years data (if present)
        start_date = current - np.timedelta64(365,'D')
        stop_date = plus_30 - np.timedelta64(365,'D')
        mask = np.in1d(dates, np.arange(start_date,stop_date,dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        ## add some non-revenue features
        minus_30 = current - np.timedelta64(30,'D')
        mask = np.in1d(dates, np.arange(minus_30,current,dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean())
        eng_features['recent_views'].append(df[mask]['total_views'].mean())

    X = pd.DataFrame(eng_features)
    ## combine features in to df and remove rows with all zeros
    X.fillna(0,inplace=True)
    mask = X.sum(axis=1)>0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

    if training == True:
        ## remove the last 30 days (because the target is not reliable)
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)
    
    return(X,y,dates)

def find_dates_limits(country, data_dir=None):
    """
    Read csv ts to extract min and max dates.
    """
    if not data_dir:
        data_dir = "./data/cs-train/ts-data"

    ## format country name
    country_id = re.sub("\s+","_",country.lower())

    ## get ts file name
    ts_file = "ts-" + country_id + ".csv"

    ## read csv file
    df = pd.read_csv(os.path.join(data_dir, ts_file))
    
    return (pd.to_datetime(df.loc[0,'date']), pd.to_datetime(df.loc[df.shape[0]-1,'date']))

if __name__ == "__main__":

    ## generate all time series (all + top 10 countries)
    run_start = time.time() 
    data_dir = os.path.join(".","data","cs-train")
    print("...fetching data")

    ## recreate all time series
    ts_all = fetch_ts(data_dir,clean=True,country=None)

    ## compute running time
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))

    ## print info
    for key,item in ts_all.items():
        print(key,item.shape)