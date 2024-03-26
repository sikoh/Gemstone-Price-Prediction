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