import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error


data = pd.read_csv('dataset_final_new_features.csv')
print(data.head())

features = data[['hydrophobicity','aromaticity', 'sequence_length', 'molecular_weight']]
target = data['retention_time']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)
print('gbm model fitted')

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print('training and test data prediction')

y_pred = model.predict(X_test)
print(y_pred)

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
print(f'Training R^2 Score: {train_r2}')
print(f'Testing R^2 Score: {test_r2}')
print(f'Training Median Absolute Error: {train_median_absolute_error}')
print(f'Testing Median Absolute Error: {test_median_absolute_error}')
print(f'Training Cosine Similarity: {cosine_similarity}')




