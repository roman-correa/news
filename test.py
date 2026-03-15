import sklearn
import xgboost
import lightgbm
import catboost
import numpy
import pandas
import geopandas
import streamlit
import plotly
import optuna
import mlflow
import geopy
import requests
import rapidfuzz
for m in [sklearn, xgboost, lightgbm, catboost, numpy, pandas, geopandas, streamlit, plotly, optuna, mlflow, geopy, requests, rapidfuzz]:
    print(f'{m.__name__}: {m.__version__}')
