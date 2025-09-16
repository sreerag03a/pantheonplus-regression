import os
import sys
from src.components.handling.exceptions import CustomException
from src.components.handling.logger import logging
from src.components.handling.utils import save_obj
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline


from src.components.model.model_trainer import ModelConfig, ModelTrainer

#Initial setup

pantheonpath = 'data/Pantheon+SH0ES.dat'
covariancepath='data/Pantheon+SH0ES_STAT+SYS.cov'


@dataclass
class DataConfig:
    train_data_path:str = os.path.join('outputs','train.csv')
    test_data_path:str = os.path.join('outputs','test.csv')
    covar_train_path:str = os.path.join('outputs','C_train.')
    original_data_path:str = os.path.join('outputs','ogdata.csv')

class DataIngestion:
    def __init__(self,split_ratio): 
        '''
        Split ratio - put value between 0 and 1 - This will be the ratio the training and testing data will be split into
        If split ratio is given as 0.3 - The data will be split 70:30 into training and testing sets
        
        '''
        self.ingest_conf = DataConfig()
        self.split_ratio = split_ratio
        
    def start_ingest(self):
        '''
        This is where the data is split into sets and then saved into different files
        '''
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv(pantheonpath, sep = r'\s+')
            with open(covariancepath) as f:
                N=int(f.readline().strip())
                covmatrix=np.fromfile(f,sep=" ")
            covmatrix = covmatrix.reshape((N,N))
            logging.info(f'Covmatrix of shape {covmatrix.shape} loaded successfully')
            logging.info('Successfully ingested data as a Dataframe')

            os.makedirs(os.path.dirname(self.ingest_conf.train_data_path), exist_ok= True)
            nrow,ncol = df.shape

            df.to_csv(self.ingest_conf.original_data_path, index=False,header=True)
            logging.info("Train test split initiated - Splitting {} datapoints into {} training points and {} test points".format(nrow,round((1-self.split_ratio)*nrow),round(self.split_ratio*nrow)))
            
            train_index, test_index = train_test_split(np.arange(nrow), test_size=self.split_ratio,random_state=76)
            train_set,test_set = df.iloc[train_index,:],df.iloc[test_index,:]

            train_set.to_csv(self.ingest_conf.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingest_conf.test_data_path, index=False,header=True)
            C_train = covmatrix[np.ix_(train_index,train_index)]
            C_test = covmatrix[np.ix_(test_index,test_index)]
            self.C_train = C_train
            self.C_test = C_test
            return (
                self.ingest_conf.train_data_path,
                self.ingest_conf.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

        
@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('outputs','models','preprocessor.pkl')

class DataTransform:

    '''
    This is where the data is preprocessed. Here, the redshift is transformed with a logarithm,
    and other inputs and imputed and scaled.
    
    '''
    def __init__(self):
        self.data_transform_config = DataTransformConfig()
    
    def get_transformer_obj(self):
        try:
            other_columns = ["x1","c"]

            log_transformer = FunctionTransformer(func= lambda x: np.log10(x))

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler() )
                ]
            )        
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline, other_columns),
                ("log_transformer", log_transformer,['zHD'])
                ]
            )
            logging.info("Numerical values Imputed and Standard Scaled")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_transformer_obj()

            target_column = "m_b_corr"

            train_df_mod = train_df.drop(columns=[target_column], axis = 1)
            target_train_df = train_df[target_column]

            test_df_mod = test_df.drop(columns=[target_column], axis = 1)
            target_test_df = test_df[target_column]

            logging.info("Preprocessing data...")

            train_arr = preprocessor_obj.fit_transform(train_df_mod)
            test_arr = preprocessor_obj.transform(test_df_mod)

            train_arr = np.c_[train_arr,np.array(target_train_df)]
            test_arr = np.c_[test_arr,np.array(target_test_df)]

            logging.info("Preprocess completed")

            save_obj(

                filepath = self.data_transform_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )
            logging.info("Saved preprocessing object in outputs folder")
            return (
                train_arr,
                test_arr ,
                self.data_transform_config.preprocessor_obj_path

            )

        except Exception as e:
            logging.info("Error occured in transformation/preprocessing")
            raise CustomException(e,sys)
