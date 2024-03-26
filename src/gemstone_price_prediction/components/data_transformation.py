import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.gemstone_price_prediction.exception import CustomException
from src.gemstone_price_prediction.logger import logging
from src.gemstone_price_prediction.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation()

    def get_data_transformation_object(self):
        try:
            # Define the categorical(ordinal-encoded) and numerical(scaled) columns
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            # Create numerical pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Create categorical pipeline
            categ_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info(f'Categorical Columns : {categorical_cols}')
            logging.info(f'Numerical Columns   : {numerical_cols}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols)
                    ('categ_pipeline', categ_pipeline, categorical_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation Phase')
            raise CustomException(e, sys)