import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error

from time import time
from datetime import timedelta


X = pd.read_csv('USDA-0.2.csv')
y = X['pct_obese_adults13']

X = X[['fips', 'pct_diabetes_adults13']]

cv = KFold(n_splits=5, shuffle=True, random_state=101)


def run_model(model, data, target, cv):
    start = time()
    oof_preds = cross_val_predict(model, data, target, cv=cv, method='predict')
    end = time()
    diff = end - start
    td = timedelta(seconds=diff)
    print('Completed in {}'.format(td))
    mae = mean_absolute_error(target, oof_preds)
    return mae


models = [
    (RandomForestRegressor(n_jobs=-1, n_estimators=100, criterion='mae'), 'RandomForest'),
    (LinearRegression(), 'LinearRegression'),
    (Lasso(), 'LassoRegression'),
    (Ridge(), 'RidgeRegression')
]

for model, name in models:
    print('-'*30)
    print(f'Starting {name}')
    mae = run_model(model, X, y, cv)
    print(f'{name} MAE: {mae:.4f}')
    print('-'*30)
