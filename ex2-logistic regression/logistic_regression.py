import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())


# plot data
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'],
           s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'],
           s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


# calculate cost function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# plot the sigmoid function
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


# set X and y
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, :-1]
y = data.iloc[:, -1:cols]

X = np.array(X.values)
y = np.array(y.values).reshape(-1, 1)
theta = np.zeros(3).reshape(-1, 1)
print(f"X: {X.shape}, y: {y.shape}, theta: {theta.shape}")


def cost(theta, X, y):
    theta = theta.reshape(-1, 1)  # explicitly ensure that
    first = np.multiply(y, np.log(sigmoid(X @ theta)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X @ theta)))
    return (-1 / len(y)) * np.sum(first + second)


print(f"initial cost: {cost(theta, X, y)}")


def gradient(theta, X, y):
    theta = theta.reshape(-1, 1)  # explicitly ensure that
    error = sigmoid(X @ theta) - y
    m = y.shape[0]
    grad = (1/m) * (X.T @ error)
    return grad.flatten()  # Return 1D array for scipy


print(f"initial grad: {gradient(theta, X, y)}")


# use scipy's truncated newton (TNC) to optimize cost function
result = opt.fmin_tnc(func=cost, x0=theta.flatten(),
                      fprime=gradient, args=(X, y))
print(result)


# give out 0/1 predictions
def predict(theta, X):
    theta = theta.reshape(-1, 1)
    probability = sigmoid(X @ theta)

    return [1 if x >= 0.5 else 0 for x in probability]


predictions = predict(result[0], X)
correct = [1 if (a ^ b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print(f"accuracy is: {accuracy}!")
