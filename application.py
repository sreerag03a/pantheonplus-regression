from flask import  Flask, request, render_template
import numpy as np
import pandas as pd
import sys
from src.components.handling.logger import logging
from src.components.handling.exceptions import CustomException

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app= application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        zHD=request.form.get('zHD')
        x1 = request.form.get('x1')
        c = request.form.get('c')
        data = CustomData(
            zHD,
            x1,
            c
        )
        model = request.form.get('model')
        pred_df = data.get_dataframe()
        print(pred_df)

        pred_pipeline =  PredictPipeline()
        try: 
            logging.info('Prediction pipeline started')
            results = pred_pipeline.predictmB(pred_df,model)
            logging.info('Prediction pipeline completed')
            print(results)
            return render_template('home.html', results = results[0], prev_input = {'zHD' : zHD, 'x1' : x1, 'c': c, 'model' : model})
        except Exception as e:
            raise CustomException(e,sys)

    

if __name__ == '__main__':
    app.run(host='0.0.0.0')