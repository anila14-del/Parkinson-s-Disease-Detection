import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
try:
    # Use read_csv instead of read_excel since it's a .csv file
    df = pd.read_csv(r"C:\Users\acer\Downloads\parkinsons\parkinsons_data.csv")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Verify 'status' column presence
if 'status' not in df.columns:
    print("The 'status' column is missing from the dataset.")
    exit()

# Separate features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]  # excluding first ID column
labels = df.loc[:, 'status'].values
print(f"Number of positive cases (1): {np.sum(labels == 1)}")
print(f"Number of negative cases (0): {np.sum(labels == 0)}")

# Scaling features to the range [-1, 1]
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Model training
model = XGBClassifier()
model.fit(x_train, y_train)

# Predictions and accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
