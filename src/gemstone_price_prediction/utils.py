import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.gemstone_price_prediction.exception import CustomException
from src.gemstone_price_prediction.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(Xtrain, ytrain, Xtest, ytest, models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(Xtrain, ytrain)

            # Predict Testing data
            y_test_pred = model.predict(Xtest)

            # Get R2 score for test data
            test_model_score = r2_score(ytest, y_test_pred)
            report[model_name] = test_model_score

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
        

def model_metrics(true, predicted):
    try :
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)
    

def print_evaluated_results(Xtrain,ytrain,Xtest,ytest,model):
    try:
        y_train_pred = model.predict(Xtrain)
        y_test_pred = model.predict(Xtest)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = model_metrics(ytrain, y_train_pred)
        model_test_mae , model_test_rmse, model_test_r2 = model_metrics(ytest, y_test_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)