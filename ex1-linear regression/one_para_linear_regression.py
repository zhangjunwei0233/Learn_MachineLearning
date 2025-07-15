import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "ex1data1.txt"
data = pd.read_csv(path, header=None, names=["Population", "Profit"])

# quick look of the data
# print(data.head())
# print(data.describe())
# data.plot(kind="scatter", x="Population", y="Profit", figsize=(12, 8))
# plt.show()


data.insert(0, "Ones", 1)  # insert a colomn of data named "ones" at colomn 0
# print(data.head())

# set X (training data) and y (target value)
cols = data.shape[1]  # returns nr_row and nr_col in tuple
X = data.iloc[:, :-1]  # get the first tow cols
y = data.iloc[:, cols - 1: cols]  # get the last col

# convert to matrixs
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))


# compute cost function
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


print(f"initial cost:\t{computeCost(X, y, theta)}")


# gradientDescend method to calculate theta
def gradientDescend(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):  # for each step
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

# normal equation method to calculate theta


def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


alpha = 0.01
iters = 1000
g, cost = gradientDescend(X, y, theta, alpha, iters)
print(f"final cost:\t{cost[-1]}")
print(f"final result:\t{g}")


# draw out the predicted line
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label="Traning Data")
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# draw out cost reduction history
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

print(f"using normal equation: {normalEqn(X, y)}")
