from flask import  Flask, request, render_template, send_file
import numpy as np
import pandas as pd
import sys
from src.components.handling.logger import logging
from src.components.handling.exceptions import CustomException

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import io

application = Flask(__name__)

app= application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    pred_pipeline =  PredictPipeline()
    if request.method =='GET':
        return render_template('home.html')
    elif request.method == 'POST':
        file = request.files.get('file')
        model = request.form.get('model')

        if file and file.filename.strip() != '':
            logging.info('Reading file...')
            try:
                pred_df = pd.read_csv(file)
                results = pred_pipeline.predictmB(pred_df,model)
                pred_df['m_b_pred'] = results

                output = io.StringIO()
                pred_df.to_csv(output, index=False)
                output.seek(0)

                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='predictions.csv'
                )
            except Exception as e:
                raise CustomException(e,sys)
        else:
            zHD=float(request.form.get('zHD'))
            x1 = request.form.get('x1')
            c = request.form.get('c')
            data = CustomData(
                zHD,
                x1,
                c
            )
            pred_df = data.get_dataframe()
            print(pred_df)
            try: 
                logging.info('Prediction pipeline started')
                results = pred_pipeline.predictmB(pred_df,model)
                logging.info('Prediction pipeline completed')
                print(results)
                return render_template('home.html', results = results[0], prev_input = {'zHD' : zHD, 'x1' : x1, 'c': c, 'model' : model})
            except Exception as e:
                raise CustomException(e,sys)
    
            
@app.route('/predictmbwitherr', methods = ['GET', 'POST'])
def predict_datapoint_witherr():
    pred_pipeline =  PredictPipeline()
    if request.method =='GET':
        return render_template('predwitherr.html')
    elif request.method == 'POST':
        file = request.files.get('file')
        model = request.form.get('model')

        if file and file.filename.strip() != '':
            logging.info('Reading file...')
            try:
                pred_df = pd.read_csv(file)
                results = pred_pipeline.predictmBwitherr(pred_df,model)
                pred_df['m_b_pred'] = results[0]
                pred_df['m_b_err'] = results[1]

                output = io.StringIO()
                pred_df.to_csv(output, index=False)
                output.seek(0)

                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='predictions.csv'
                )
            except Exception as e:
                raise CustomException(e,sys)
        else:
            zHD=float(request.form.get('zHD'))
            x1 = request.form.get('x1')
            c = request.form.get('c')
            data = CustomData(
                zHD,
                x1,
                c
            )
            pred_df = data.get_dataframe()
            print(pred_df)
            try: 
                logging.info('Prediction pipeline started')
                results = pred_pipeline.predictmBwitherr(pred_df,model)
                logging.info('Prediction pipeline completed')
                print(results)
                return render_template('predwitherr.html', results = results, prev_input = {'zHD' : zHD, 'x1' : x1, 'c': c, 'model' : model})
            except Exception as e:
                raise CustomException(e,sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)