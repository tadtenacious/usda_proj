import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error

X = pd.read_csv('USDA-0.2.csv')
y = X['pct_obese_adults13']

drop_cols = ['fips','pct_diabetes_adults13','pct_diabetes_adults08']
X.drop(drop_cols, axis=1)

scorer = make_scorer(mean_absolute_error)

models = [
    ('RandomForrest', RandomForestRegressor(n_jobs=-1, n_estimators=100, criterion='mae')),
    ('LinearRegression', LinearRegression()),
    ('LassoRegression', Lasso()),
    ('RidgeRegression', Ridge())
]
cv = KFold(n_splits=5, shuffle=True, random_state=101)

for name, model in models:
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer,n_jobs=-1)
    print('{} Average MAE: {:.2f}'.format(name, np.mean(scores)))