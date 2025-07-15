import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'ex1data2.txt'
data = pd.read_csv(path, header=None, names=["Size", "Bedrooms", "Price"])

# standardize
data = (data - data.mean()) / data.std()

# create X, Y and theta matrix
data.insert(0, "Ones", 1)

nr_cols = data.shape[1]
X = data.iloc[:, :-1]
Y = data.iloc[:, nr_cols - 1:nr_cols]

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.zeros((X.shape[1], 1))


# calculate J and steps
def computeCost(X, Y, theta):
    inner = (X @ theta) - Y
    return ((inner.T @ inner) / (2 * Y.shape[0])).item()


def gradientDescent(X, Y, theta, alpha, iters):
    costs = np.zeros(iters)

    for i in range(iters):
        error = (X @ theta) - Y
        step = (alpha / Y.shape[0]) * (X.T @ error)
        theta = theta - step

        costs[i] = computeCost(X, Y, theta)
    return theta, costs


# perform regression
alpha = 0.01
iters = 1000
g, costs = gradientDescent(X, Y, theta, alpha, iters)
print(f"final cost:\t{costs[-1]}")

# plot result
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), costs, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
