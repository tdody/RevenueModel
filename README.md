# IBM AI Enterprise Workflow Capstone
Files for the IBM AI Enterprise Workflow Capstone project. 

## API
===========
  
The API contains four tabs:
1. The **Home** tab is used to show how to use the API.
2. The **Train** tab is used to re-train a model.
3. The **Predict** tab is used to make revenue predictions.
4. The **Logs** tab is used to display the content of the train and predict logs.

## Usage
===========
  
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