import os
import sys
import numpy as np
import pandas as pd
from src.components.handling.exceptions import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from src.components.handling.logger import logging

def save_obj(filepath,obj):

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as fileobj:
            dill.dump(obj, fileobj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,eval_metric, params):

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
            # model_score = [j(y_test,y_preds) for j in eval_metric]

            report[list(models.keys())[i]] = model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(filepath):
    try:
        with open(filepath, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)