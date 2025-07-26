import sys
import pandas as pd

from src.components.handling.exceptions import CustomException
from src.components.handling.logger import logging
from src.components.handling.utils import load_obj

class PredictPipeline:

    def __init__(self):
        pass

    def predictmB(self,features,selected_model = None):
        try:
            if selected_model == 'Best Model':
                model_path = 'outputs/models/trained_model.pkl'
            else:
                model_path = f'outputs/models/{selected_model}.pkl'
            preprocessor_path = 'outputs/models/preprocessor.pkl'

            logging.info('Loading model...')
            model = load_obj(model_path)

            logging.info('Loading preprocessor for data...')
            preprocessor = load_obj(preprocessor_path)

            logging.info('Attempting preprocessor')
            data_scaled = preprocessor.transform(features)
            logging.info('Completed preprocessing - {}'.format(data_scaled))
            res = model.predict(data_scaled)
            return res
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, zHD:float, x1:float, c: float):
        self.zHD = zHD
        self.x1 = x1
        self.c = c

    def get_dataframe(self):
        try:
            data_dict = {
                'zHD' : [self.zHD],
                'x1' : [self.x1],
                'c' : [self.c]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            return CustomException(e,sys)
