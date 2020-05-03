# IBM AI Enterprise Workflow Capstone
Files for the IBM AI Enterprise Workflow Capstone project. 

## API
  
The API contains four tabs:
1. The **Home** tab is used to show how to use the API.
2. The **Train** tab is used to re-train a model.
3. The **Predict** tab is used to make revenue predictions.
4. The **Logs** tab is used to display the content of the train and predict logs.

## Checklist

### EDA
Did the EDA investigation use visualizations?  
    - Done in `./notebooks/EDA.ipynb`  

Does the data ingestion exists as a function or script to facilitate automation?  
    - Done using `engineer_features()` from `./cslib.py`  

### Test
Are there unit tests for the API?  
    - Done using `./unittests/ApiTests.py`  

Are there unit tests for the model?  
    - Done using `./unittests/ModelTests.py`  

Are there unit tests for the logging?  
    - Done using `./unittests/LoggerTests.py`  

Can all of the unit tests be run with a single script and do all of the unit tests pass?  
    - Done using `./run-tests.py`  

Was there an attempt to isolate the read/write unit tests from production models and logs?  
    - Logs: `./unittests/LoggerTests.py`  
    - Model: `./unittests/ModelTests.py`  

### Performance monitoring
Is there a mechanism to monitor performance?  
    - Done using `./monitoring.py`  

### API
Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?  
    - Build docker image, run docker container.  
    - Use Predict tab for prediction.  
    - Enter the country and the date:  
        - For example France, 01/05/2018  

Were multiple models compared?  
    - Done using `./notebooks/Part_2_Models.ipynb`  

Is everything containerized within a working Docker image?
    - Done using `./Dockerfile`  
    - Run: `docker build -t revenue-model .`  
    - Run: `docker run -it -p 8080:8080 revenue-model`  

Did they use a visualization to compare their model to the baseline model?  
    - Done using `./notebooks/Part_2_Models.ipynb`  

## Usage
  
To test app.py
--------------

``` {.bash}
~$ python app.py
```

Go to <http://0.0.0.0:8080/> and you will see a basic website that can be customtized for a project.

To test the model directly
--------------------------

see the code at the bottom of `model.py`

``` {.bash}
~$ python model.py
```

To build the docker container
-----------------------------

``` {.bash}
~$ docker build -t revenue-model .
```

Check that the image is there.

``` {.bash}
~$ docker image ls
```

Run the unittests
-----------------

To run only the api tests

``` {.bash}
~$ python unittests/ApiTests.py
```

To run only the model tests

``` {.bash}
~$ python unittests/ModelTests.py
```

To run only the logg tests

``` {.bash}
~$ python unittests/LoggerTests.py
```

To run all of the tests

``` {.bash}
~$ python run-tests.py
```

Run the container to test that it is working
--------------------------------------------

``` {.bash}
~$ docker run -it -p 8080:8080 revenue-model
```

Go to <http://0.0.0.0:4000/> to make predictions, re-train the models.
