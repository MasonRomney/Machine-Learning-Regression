# Inporting libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train the polynomial model on the datset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 4)
X_poly = polynomial.fit_transform(X)
linear = LinearRegression()
linear.fit(X_poly, y)

# Visualize the results
plt.scatter(X, y, color='red')
plt.plot(X, linear.predict(X_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Higher resolution graph
n = 0.1 # step size
X_grid = np.arange(X.min(), X.max(), n)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_poly = polynomial.transform(X_grid)
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear.predict(X_grid_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
