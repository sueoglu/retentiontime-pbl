import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("peptides_with_features.csv")

feature_cols = [
    "gravy",
    "fraction_ILVFW",
    "aliphatic_index",
    "sequence_length_x_aromaticity"
]

X = df[feature_cols]
y = df["retention_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

for col, coef in zip(feature_cols, model.coef_):
    print(f"{col}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
