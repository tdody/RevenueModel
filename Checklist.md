# Checklist

## EDA
Did the EDA investigation use visualizations?
    Done in `./notebooks/EDA.ipynb`

Does the data ingestion exists as a function or script to facilitate automation?
    Done using `engineer_features()` from `./cslib.py`

## Test
Are there unit tests for the API?
    Done using `./unittests/ApiTests.py`

Are there unit tests for the model?
    Done using `./unittests/ModelTests.py`

Are there unit tests for the logging?
    Done using `./unittests/LoggerTests.py`

Can all of the unit tests be run with a single script and do all of the unit tests pass?
    Done using `./run-tests.py`

Was there an attempt to isolate the read/write unit tests from production models and logs?
    Logs: `./unittests/LoggerTests.py`
    Model: `./unittests/ModelTests.py`

## Performance monitoring
Is there a mechanism to monitor performance?
    Done using `./monitoring.py`

## API
Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?
    Build docker image, run docker container.
    Use Predict tab for prediction.
    Enter the country and the date:
        For example France, 01/05/2018

Were multiple models compared?
    Done using `./notebooks/Part_2_Models.ipynb`

Is everything containerized within a working Docker image?
    Done using `./Dockerfile`
    Run: `docker build -t revenue-model .`
    Run: `docker run -it -p 8080:8080 revenue-model`

Did they use a visualization to compare their model to the baseline model?
    Done using `./notebooks/Part_2_Models.ipynb`