# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# training the decistion tree model
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state = 0)
tree.fit(X, y)

# predict a new result
tree.predict([[5]])

# visualizing the results
n = 0.1
X_grid = np.arange(X.min(), X.max(), n)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, tree.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
