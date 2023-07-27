import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MLR
from polynomial_model import add_polynomial_features
from data_spliter import data_spliter
from search_alpha_max_iter import search_alpha_max_iter as sami

if __name__ == "__main__":
    