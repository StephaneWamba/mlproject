import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

import dill

def save_object(file_path, obj):
    '''
        This function will save the object to the file path
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models:dict, params:dict, cv:int=3, n_jobs:int=-1, verbose:int=0, refit:bool=True):
    '''
        This function will evaluate the model and return the model report
    '''
    try:
        model_report = {}
        for model_name, model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            model_report[model_name] = r2_score(y_test, y_pred)
        return model_report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    '''
        This function will load the object from the file path
    '''
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)