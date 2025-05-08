import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import randint

data = pd.read_csv('/Users/oykusuoglu/PythonProjects/RetentionTimePBL/normalized_pw_dataset.csv')
print('Data loaded')

#data_sampled = data.sample(frac=0.2, random_state = 42)


X = data[['gravy','aliphatic_index', 'count_hydrophobic','count_aromatic',
          'fraction_ILVFW', 'sequence_length_x_aromaticity']]
y = data['retention_time']
print('Data in input and output loaded')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('training and test data split')


#param_dist = {
#   'n_estimators': randint(26, 38),
#    'max_depth': randint(29, 45),
#    'min_samples_split': randint(10, 22),
#    'min_samples_leaf': randint(8, 18),
    #'subsample': [0.8, 0.9, 1.0],
    #'max_features': ['auto', 'sqrt']
#}

# Step 4: Train the Gradient Boosting Regressor
gbm = GradientBoostingRegressor(
    random_state=42,
    max_depth= 49,
    n_estimators= 46,
    min_samples_split=11,
    min_samples_leaf=14
    )
print('hyperparameters')

#random_search = RandomizedSearchCV(
#    estimator=gbm,
#    param_distributions=param_dist,
#    n_iter=3,
#    cv=2,
#    n_jobs=-1,
#    verbose=2,
#    scoring='neg_mean_squared_error'
#)

#print("Starting grid search...")
#random_search.fit(X_train, y_train)

#print("\nBest parameters found:")
#print(random_search.best_params_)
#print(f"\nBest cross-validation RMSE: {np.sqrt(-grid_search.best_score_)}")

#best_gbm = random_search.best_estimator_

gbm.fit(X_train, y_train)
print('gbm model fitted')

y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)
print('training and test data prediction')

y_pred = gbm.predict(X_test)
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

