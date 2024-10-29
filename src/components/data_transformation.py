import os
import sys

from ..logger import logging
from ..exception import CustomException

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

from dataclasses import dataclass
@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer(self):
        '''
            This function will return the preprocessor object which will be used to transform the data
        
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender',
                                   'race_ethnicity',
                                   'parental_level_of_education',
                                      'lunch',
                                        'test_preparation_course']
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            logging.info('Numerical and Categorical pipelines have been created')
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numerical_columns),
                ('cat', cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_tranformation(self, train_data_path, test_data_path):
        '''
            This function will initiate the data transformation process
        '''
        try:
            logging.info('Data transformation has started')
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            preprocessor = self.get_data_transformer()
            logging.info('Read train and test data successfully')

            target_column_name = 'math_score'
            

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'math_score']

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)  

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            logging.info('Saved preprocessing object')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e: 
            raise CustomException(e, sys)
            