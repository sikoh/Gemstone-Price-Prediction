import sys
import os
from dataclasses import dataclass

# Basic Import
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# internal imports
from src.gemstone_price_prediction.exception import CustomException
from src.gemstone_price_prediction.logger import logging
from src.gemstone_price_prediction.utils import save_object, evaluate_models, model_metrics, print_evaluated_results
