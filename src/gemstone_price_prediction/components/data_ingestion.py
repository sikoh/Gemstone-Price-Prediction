import os
import sys
from src.gemstone_price_prediction.exception import CustomException
from src.gemstone_price_prediction.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.gemstone_price_prediction.components.data_transformation
# from src.gemstone_price_prediction.components.model_trainer