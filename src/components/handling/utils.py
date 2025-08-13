import os
import sys
import numpy as np
import pandas as pd
from src.components.handling.exceptions import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from src.components.handling.logger import logging
import urllib.request

'''
Different utility functions used throughout the project
'''

def save_obj(filepath,obj):
    # To pickle models or fitted preprocessor

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as fileobj:
            dill.dump(obj, fileobj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,eval_metric, params):
    #To tune different models based on eval_metric
    try:
        report = {}
        for i in range(len(list(models))):
            logging.info('Evaluating {}'.format(list(models.keys())[i]))
            model = list(models.values())[i]

            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv = 3)

            gs.fit(X_train, y_train)


            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_preds = model.predict(X_test)

            model_score = eval_metric(y_test,y_preds)

            report[list(models.keys())[i]] = model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(filepath):
    #Load pickled object
    try:
        with open(filepath, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    


def download_data(datadir):
    # Download the required Pantheon+ and DES data
    dwnld_file = {
    "Pantheon+SH0ES.dat" : "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "DES-data.csv" : "https://raw.githubusercontent.com/des-science/DES-SN5YR/main/4_DISTANCES_COVMAT/DES-SN5YR_HD%2BMetaData.csv"

    }
    for filename, url in dwnld_file.items():
        save_path = os.path.join(datadir, filename)

        logging.info(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, save_path)
            logging.info(f"Saved to {save_path}")
        except Exception as e:
            raise CustomException(e,sys)