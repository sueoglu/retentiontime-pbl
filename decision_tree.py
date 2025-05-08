import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = pd.read_csv('final_grouped_t_stard.csv')
print(data.head())
print(data.shape)

df = pd.DataFrame(data)

#data['sequence_length'] = data['sequence'].apply(len)
#print(data.head())

#label_encoder = LabelEncoder()
#df['sequence_encoded'] = label_encoder.fit_transform(df['sequence'])

onehot_encoder = OneHotEncoder()
sequence_encoded = onehot_encoder.fit_transform(df[['sequence']]).toarray()

sequence_encoded_df = pd.DataFrame(sequence_encoded, columns=onehot_encoder.get_feature_names_out(['sequence']))

features = pd.concat([sequence_encoded_df, df[['hydrophobicity', 'aromaticity', 'tyrosine_count']]], axis=1)
target = df['retention_time']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=1)

model = DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

model.fit(X_train, y_train)

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R2: {train_r2}')
print(f'Testing R2: {test_r2}')



