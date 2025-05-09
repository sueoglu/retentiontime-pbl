import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv('grouped_cystein5_feature_ex.csv')

print(data.head())

X = data[['hydrophobicity','seq_length', 'molecular_weight', 'instability_index', 'isoelectric_point', 'cystein_count','aromaticity']]
y = data['retention_time']    # Target: retention time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)

y_test_pred = rf.predict(X_test)

y_pred = rf.predict(X_test)
print(y_pred)

y_test_2D = np.array(y_test).reshape(1,-1)
y_pred_2D = np.array(y_pred).reshape(1,-1)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_median_absolute_error = median_absolute_error(y_train, y_train_pred)
test_median_absolute_error = median_absolute_error(y_test, y_test_pred)

cosine_similarity = cosine_similarity(y_test_2D, y_pred_2D)

print(f'Training Mean Absolute Error: {train_mae}')
print(f'Testing Mean Absolute Error: {test_mae}')
print(f'Training R^2 Score: {train_r2}')
print(f'Testing R^2 Score: {test_r2}')
print(f'Training Median Absolute Error: {train_median_absolute_error}')
print(f'Testing Median Absolute Error: {test_median_absolute_error}')
print(f'Training Cosine Similarity: {cosine_similarity}')


