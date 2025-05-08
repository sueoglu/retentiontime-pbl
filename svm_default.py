import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('dataset_paper_final.csv')
print('CSV loaded')


X = data[['hydrophobicity', 'sequence_length', 'molecular_weight', 'instability_index', 'isoelectric_point', 'cystein_count', 'aromaticity']]
y = data['retention_time']
print('Features and target extracted')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Data split into training and testing sets')

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
print('Indices reset')

X_train_sample = X_train
y_train_sample = y_train

print('Random sample of training data selected')

scaler = StandardScaler()
X_train_scaled_sample = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test)
print('Feature scaling applied')

svr = SVR(kernel='rbf', max_iter=10000000)
svr.fit(X_train_scaled_sample, y_train_sample)

y_train_pred = svr.predict(X_train_scaled_sample)
y_test_pred = svr.predict(X_test_scaled)

y_train_pred = svr.predict(X_train_scaled_sample)
y_test_pred = svr.predict(X_test_scaled)
print('Predictions made')

train_mse = mean_squared_error(y_train_sample, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_sample, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train_sample, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Model Performance:")
print(f'Training Mean Squared Error (MSE): {train_mse}')
print(f'Testing Mean Squared Error (MSE): {test_mse}')
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

#svr = SVR(kernel='rbf', max_iter=10000000)  # Use 'rbf' kernel with a limit on iterations
#svr.fit(X_train_scaled_sample, y_train_sample)
#print('Model trained')

#y_train_pred = svr.predict(X_train_scaled_sample)
#y_test_pred = svr.predict(X_test_scaled)
#print('Predictions made')

#train_mse = mean_squared_error(y_train_sample, y_train_pred)
#test_mse = mean_squared_error(y_test, y_test_pred)
#train_r2 = r2_score(y_train_sample, y_train_pred)
#test_r2 = r2_score(y_test, y_test_pred)


#train_rmse = np.sqrt(train_mse)
#test_rmse = np.sqrt(test_mse)


#print(f'Training Mean Squared Error (MSE): {train_mse}')
#print(f'Testing Mean Squared Error (MSE): {test_mse}')
#print(f'Training Root Mean Squared Error (RMSE): {train_rmse}')
#print(f'Testing Root Mean Squared Error (RMSE): {test_rmse}')
#print(f'Training R^2 Score: {train_r2}')
#print(f'Testing R^2 Score: {test_r2}')