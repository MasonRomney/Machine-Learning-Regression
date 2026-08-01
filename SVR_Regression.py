# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
y_scaler = StandardScaler()
scaled_X = X_scaler.fit_transform(X)
scaled_y = y_scaler.fit_transform(y.reshape(-1,1)).ravel()

# Train the SVR Model (rbf kernel)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(scaled_X, scaled_y)

# Predict a new result
scaled_input = X_scaler.transform([[5]])
scaled_prediction = regressor.predict(scaled_input).reshape(-1,1)
original_prediction = y_scaler.inverse_transform(scaled_prediction)
print(original_prediction)

# Visualize results with pyplot
scaled_y_pred = regressor.predict(scaled_X).reshape(-1,1)
y_pred = y_scaler.inverse_transform(scaled_y_pred)
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('SVR Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Visualize results wtih higher resolution and smoother curve
X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
scaled_X_grid = X_scaler.transform(X_grid)
scaled_y_pred_grid = regressor.predict(scaled_X_grid).reshape(-1,1)
y_pred_grid = y_scaler.inverse_transform(scaled_y_pred_grid)
plt.scatter(X, y, color='red')
plt.plot(X_grid, y_pred_grid, color='blue')
plt.title('SVR Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
