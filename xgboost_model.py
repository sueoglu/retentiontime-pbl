import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

data = pd.read_csv('dataset_paper_final.csv')

X = data[['hydrophobicity', 'aromaticity', 'isoelectric_point', 'molecular_weight', 'sequence_length', 'instability_index', 'cystein_count']]
y = data['retention_time']
print('Data in input and output loaded')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('training and test data split')

xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=27,
                             learning_rate=0.3821,
                             max_depth=15,
                             gamma=0.0867,
                             reg_alpha=0.7,
                             reg_lambda=18)
print("Initialize the XGBoost regressor")

#random_search.fit(X_train, y_train)


xgb_model.fit(X_train, y_train)
print("Train XGBoost model")

#print("Best parameters:", random_search.best_params_)

#best_model = random_search.best_estimator_

y_train_pred =xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)
y_pred = xgb_model.predict(X_test)

#y_train_pred = xgb_model.predict(X_train)
#y_test_pred = xgb_model.predict(X_test)
#y_pred = xgb_model.predict(X_test)

y_test_2D = np.array(y_test).reshape(1,-1)
y_pred_2D = np.array(y_pred).reshape(1,-1)


train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
cosine_similarity = cosine_similarity(y_test_2D, y_pred_2D)
print('errors calculated')

train_median_absolute_error = median_absolute_error(y_train, y_train_pred)
test_median_absolute_error = median_absolute_error(y_test, y_test_pred)


print(f'Training Mean Absolute Error: {train_mae}')
print(f'Testing Mean Absolute Error: {test_mae}')
print(f'Training R2 Score: {train_r2}')
print(f'Testing R2 Score: {test_r2}')
print(f'Training Median Absolute Error: {train_median_absolute_error}')
print(f'Testing Median Absolute Error: {test_median_absolute_error}')
print(f'Training Cosine Similarity: {cosine_similarity}')

mse = mean_squared_error(y_test, y_pred)
print(f"MSE after tuning: {mse}")