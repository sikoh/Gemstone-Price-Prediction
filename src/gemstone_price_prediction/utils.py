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