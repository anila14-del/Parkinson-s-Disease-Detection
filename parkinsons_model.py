# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:06:00 2024

@author: acer
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to train the model
def train_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Calculated accuracy: {accuracy * 100:.2f}%")  # Display accuracy as percentage
    return accuracy

# Load the dataset
try:
    df = pd.read_csv(r"C:\Users\acer\Downloads\parkinsons\parkinsons_data.csv")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Verify 'status' column presence
if 'status' not in df.columns:
    print("The 'status' column is missing from the dataset.")
    exit()

# Separate features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]  # Excluding first ID column
labels = df['status'].values

# Output the count of positive and negative cases
print(f"Number of positive cases (1): {np.sum(labels == 1)}")
print(f"Number of negative cases (0): {np.sum(labels == 0)}")

# Scaling features to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(features)
y = labels

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train the model using the train_model function
model = train_model(x_train, y_train)

# Evaluate the model using the evaluate_model function and store the result
test_accuracy = evaluate_model(model, x_test, y_test)

# Optionally, print the test accuracy after evaluation
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
