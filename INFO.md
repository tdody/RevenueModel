# STRUCTURE

## DATA

The data is stored in ./data
Two folders:
 - cs-production contains invoices for 2019 (08 to 12)
 - cs-train contains invoices from 2017-11 through 2019-07
 - Content is .json

## PYTHON

### cslib.py
 - fetches data from folder `./cs-production`
 - convert data to time-series
    - for all countries
    - for individual countries

 - `fetch_data(data_dir)`
    - Collect the data stored into json files.
    - Return a pandas DataFrame containing the combined json files.
    
 - `convert_to_ts(df_orig, country=None)`
    - takes original dataframe as input
    - filter the data by country (if specified)
    - return aggregated time series over each day

 - `fetch_ts(data_dir, clean=False)`
    - convenience function to read in new data
    - uses csv to load quickly
    - use clean=True when you want to re-create the files
    - returns dictionary of dataFrames {country: df}

 - `engineer_features(df,training=True)`
    - for any given day the target becomes the sum of the next 30 days revenue
    - for that day we engineer several features that help predict the summed revenue
    - the 'training' flag will trim data that should not be used for training
    - when set to false all data will be returned
    - engineered features:
        - *previous_N*: sum of the revenue over the last N days
        - *previous_year*: sum of the revenue over 30 days (one year before)
        - *recent_invoices*: mean number of invoices over last 30 days
        - *recent_views*: mean number of views over last 30 days

    - returns:
        - X: dataframe of data
        - y: sum of the revenue during the 30 days following `date`
        - dates: dates covered by X

### model.py

- `_model_train(df,tag,test=False)`
    - The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 
    - base model is random forest
    - saves model in directory and update log

- `model_load(prefix='sl',data_dir=None,training=True)`
    - the prefix allows the loading of different models
    - return all_data, all_model
        - all_data:
            - X: full data frame
            - y: revenue
            - dates: scope of input

- `model_predict(country,year,month,day,all_models=None,test=False)`
    - if no model then load all models
    - 