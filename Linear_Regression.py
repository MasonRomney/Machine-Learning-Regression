# Inporting libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Training the regression model
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)

# Predicting the test set results
y_pred = linear.predict(X_test)

# Visualizing the training test set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linear.predict(X_train), color='blue')
plt.title('Training set')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Visualizing the test test set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linear.predict(X_train), color='blue')
plt.title('Test set')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Print the slope and intercept
print(linear.coef_)
print(linear.intercept_)

# Predict a result
print(linear.predict([[5]]))
