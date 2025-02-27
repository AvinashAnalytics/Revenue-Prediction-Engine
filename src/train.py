# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('data\raw\revenue_prediction.csv')

# Basic data inspection
print("\nFirst 5 rows:")
print(df.head())
print("\nData Summary:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())

# Handle missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Convert categorical features to numeric
df['Franchise'] = df['Franchise'].astype('category').cat.codes
df['Category'] = df['Category'].astype('category').cat.codes

# Create interaction features
df['Order_Item_Ratio'] = df['Order_Placed'] / df['No_Of_Item']

# Split data into features and target variable
X = df[['Franchise', 'Category', 'No_Of_Item', 'Order_Placed', 'Order_Item_Ratio']]
y = df['Revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, '../models/scaler.pkl')

# Train and save the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, '../models/linear_regression_model.pkl')

# Train and save the Neural Network model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)
nn_model.save('../models/neural_network_model.h5')

# Evaluate models
lr_predictions = lr_model.predict(X_test_scaled)
nn_predictions = nn_model.predict(X_test_scaled).flatten()

# Linear Regression Model Evaluation
print('Linear Regression MAE:', mean_absolute_error(y_test, lr_predictions))
print('Linear Regression MSE:', mean_squared_error(y_test, lr_predictions))
print('Linear Regression R²:', r2_score(y_test, lr_predictions))

# Neural Network Model Evaluation
print('Neural Network MAE:', mean_absolute_error(y_test, nn_predictions))
print('Neural Network MSE:', mean_squared_error(y_test, nn_predictions))
print('Neural Network R²:', r2_score(y_test, nn_predictions))

# Visualize model performance comparison
model_performance = pd.DataFrame({
    'Model': ['Linear Regression', 'Neural Network'],
    'MAE': [mean_absolute_error(y_test, lr_predictions), mean_absolute_error(y_test, nn_predictions)],
    'MSE': [mean_squared_error(y_test, lr_predictions), mean_squared_error(y_test, nn_predictions)],
    'R²': [r2_score(y_test, lr_predictions), r2_score(y_test, nn_predictions)]
})

model_performance.plot(x='Model', y=['MAE', 'MSE', 'R²'], kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Error / Score')
plt.show()
