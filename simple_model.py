import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from time import time
from datetime import timedelta


X = pd.read_csv('USDA-0.2.csv')
y = X['pct_obese_adults13']

X = X[['fips','pct_obese_adults13']]

cv = KFold(n_splits=5, shuffle=True, random_state=101)

rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, criterion='mae')

start = time()
oof_preds = cross_val_predict(rf, X, y, cv=cv, method='predict')
end = time()
diff = end - start
td = timedelta(seconds=diff)
print('Completed in {}'.format(td))
mae = mean_absolute_error(y, oof_preds)
print('MAE with only two columns that I believe should be taken out: {:.4f}'.format(mae))