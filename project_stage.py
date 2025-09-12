from src.components.handling.logger import logging
from src.components.handling.exceptions import CustomException  
from src.components.handling.utils import download_data

from src.components.model.data_transformation import DataIngestion, DataTransform
from src.components.model.model_trainer import ModelTrainer, AdvancedModelTrainer
import os
import sys
import numpy as np


DATA_DIR = os.path.join(os.getcwd(),"data")
os.makedirs(DATA_DIR,exist_ok=True)

if __name__ == "__main__":
    try:
        logging.info("Attempting to download supernovae data")
        print("Check outputs/logs for logs if something goes wrong")
        download_data(DATA_DIR)
        logging.info("Downloaded Pantheon+ and DES supernovae data")
        mess1 = 'Splitting data into training and test sets in 70:30 split'
        logging.info(mess1)
        print(mess1)
        obj = DataIngestion(0.3)
        train_data,test_data = obj.start_ingest_advanced()

        data_transformation = DataTransform()

        train_arr,test_arr,_ = data_transformation.start_transform(train_data,test_data)
        print(train_arr.shape)

        modelTrainer1 = ModelTrainer()
        modelTrainer2 = ModelTrainer()

        advmodel1 = AdvancedModelTrainer()
        advmodel2 = AdvancedModelTrainer()
        cvm = np.diag(obj.C_train)
        print(cvm.shape)

        advmodel1.start_trainer(train_arr,test_arr,covmatrix=cvm,train_all=True)
        advmodel2.start_trainer(train_arr,test_arr,covmatrix=cvm,train_all=False)

        modelTrainer1.start_trainer(train_arr,test_arr, train_all=True)
        logging.info("Trained all models with hyperparameters tuned")

        modelTrainer2.start_trainer(train_arr,test_arr, train_all=False)

        logging.info("Best model found and hyperparameters tuned - Saved as trained_model.py")

        print("The project was prepared successfully - You can now run app.py and access 127.0.0.1:5000 to predict the apparent magnitude of Supernovae from input")

    except Exception as e:
        raise CustomException(e,sys)