import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error

from time import time
from datetime import timedelta

X = pd.read_csv('USDA-0.2.csv')
y = X['pct_obese_adults13']

drop_cols = ['fips','pct_diabetes_adults13','pct_diabetes_adults08']
X.drop(drop_cols, axis=1, inplace=True)

scorer = make_scorer(mean_absolute_error)

models = [
    ('RandomForrest', RandomForestRegressor(n_jobs=-1, n_estimators=100, criterion='mae')),
    ('LinearRegression', LinearRegression()),
    ('LassoRegression', Lasso(max_iter=100000)),
    ('RidgeRegression', Ridge(max_iter=100000))
]
cv = KFold(n_splits=5, shuffle=True, random_state=101)

for name, model in models:
    print('Started {}'.format(name))
    cv_start = time()
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer,n_jobs=-1)
    cv_end = time()
    cv_diff = timedelta(seconds=cv_end-cv_start)
    print('-'*30)
    print('{} Average MAE: {:.6f}'.format(name, np.mean(scores)))
    print('Completed in {}'.format(cv_diff))
    print('-'*30)