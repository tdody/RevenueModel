from flask import Flask, render_template, request, redirect
from model import *
from cslib import *
from logger import *
import pandas as pd
import argparse

## create flask app
app = Flask(__name__)


## get country names
COUNTRIES = pd.read_csv('./data/cs-train/ts-data/ts-log.csv', header=None)[1].values.tolist()

## home page
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Retrain model based on user inputs
    """
    if request.method == "GET":
        return render_template('train.html')
    else:
         ## extract field features
        country = request.form["country"]
        country_id = re.sub("\s+","_",country.lower()) 

         ## validate country
        if not country in COUNTRIES:
            output = "   Country not covered by model."
            return render_template('predict.html', country_error=output)

        ## train model
        data_dir = os.path.join('data','cs-train')
        model_train(data_dir,test=False,country=country)

        return render_template('train.html', completion_message="Model trained.") 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Make predictions based on user inputs
    """

    if request.method == "GET":
        return render_template('predict.html')
    else:

        ## extract field features
        country = request.form["country"]
        country_id = re.sub("\s+","_",country.lower()) 
        
        ## validate country
        if not country in COUNTRIES:
            output = "   Country not covered by model."
            return render_template('predict.html', country_error=output)
        
        ## retrieve dates
        date = request.form["date"]

        ## validate date format
        match = re.search(r'(\d{2}/\d{2}/\d{4})',date)
        if not match:
            output = "Format must be: MM/DD/YYYY"
            return render_template('predict.html', date_error=output)

        month, day, year = [int(x) for x in date.split("/")]
        date_dt = pd.to_datetime(date)

        ## extract min and max dates
        min_date, max_date = find_dates_limits(country)

        ## validate dates
        if min_date>date_dt or max_date<date_dt:
            output = ["Date not valid:"]
            output.append("For {0} select a date between {1} and {2}".format(country,min_date.strftime('%m/%d/%Y'), max_date.strftime('%m/%d/%Y')))
            return render_template('predict.html', prediction_text=output)

        ## load model and feature
        ## make prediction
        pred = model_predict(country_id,str(year),str(month),str(day))
        result = ["{}".format(country)]
        result.append("Revenue in the 30 days following {0}: ${1:0,.2f}".format(date, pred['y_pred'][0]))
        return render_template('predict.html', prediction_text=result)


@app.route('/logs', methods=['POST', 'GET'])
def logs():

    ## fetch training and prediction logs
    log_train = find_latest_predict_log(train=True)
    log_predict = find_latest_predict_log(train=False)

    return render_template('logs.html', 
        table_train=[log_train.to_html(classes='data')],
        table_predict=[log_predict.to_html(classes='data')])

if __name__ == "__main__":
    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=8080)