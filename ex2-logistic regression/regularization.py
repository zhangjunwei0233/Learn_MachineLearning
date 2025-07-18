# logistic regression with regularizaton
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


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

# set X and y
data.insert(0, 'Ones', 1)
data.insert(2, 'Exam 1 square', data.iloc[:, 1]**2)
data.insert(4, 'Exam 2 square', data.iloc[:, 3]**2)
print(data.head())

cols = data.shape[1]
X = data.iloc[:, :-1]
y = data.iloc[:, -1:cols]

X = np.array(X.values)
y = np.array(y.values).reshape(-1, 1)
theta = np.zeros(5).reshape(1, -1)
print(f"X: {X.shape}, y: {y.shape}, theta: {theta.shape}")


# define cost function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, lam):
    theta = theta.reshape(-1, 1)  # convert to col
    first = y * np.log(sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    penalty = lam * theta[1:]**2  # exclude theta_0 from regularization
    return (np.sum(penalty) - np.sum(first + second)) / len(y)


def gradient(theta, X, y, lam):
    theta = theta.reshape(-1, 1)
    error = sigmoid(X @ theta) - y
    m = y.shape[0]
    penalty = (lam / m) * theta
    penalty[0][0] = 0  # don't count penalty for theta_0
    grad = (1 / m) * (X.T @ error) + penalty
    return grad.flatten()


# use scipy's truncated newton (TNC) to optimize cost function
lam = 0.05
print(f"inital cost: {cost(theta, X, y, lam)}(lambda = {lam})")
print(f"initial grad: {gradient(theta, X, y, lam)}(lambda = {lam})")
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y, lam))
print(result)


# give out 0/1 predictions
def predict(theta, X):
    theta = theta.reshape(-1, 1)
    probability = sigmoid(X @ theta)

    return [1 if x >= 0.5 else 0 for x in probability]


predictions = predict(result[0], X)
correct = [1 if (a ^ b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) / len(correct))
print(f"accuracy is: {accuracy}!")

# Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Data points
plt.subplot(1, 2, 1)
admitted = data[data['Admitted'] == 1]
not_admitted = data[data['Admitted'] == 0]
plt.scatter(admitted['Exam 1'], admitted['Exam 2'],
            c='green', marker='o', label='Admitted')
plt.scatter(not_admitted['Exam 1'], not_admitted['Exam 2'],
            c='red', marker='x', label='Not Admitted')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Logistic Regression Data')
plt.legend()

# Plot 2: Cost function convergence (simplified visualization)
plt.subplot(1, 2, 2)
lambda_values = np.linspace(0.001, 1, 50)
costs = []
for lam_val in lambda_values:
    final_cost = cost(result[0], X, y, lam_val)
    costs.append(final_cost)

plt.plot(lambda_values, costs, 'b-', linewidth=2)
plt.axvline(x=lam, color='r', linestyle='--', label=f'Used λ={lam}')
plt.xlabel('Lambda (Regularization Parameter)')
plt.ylabel('Cost')
plt.title('Cost vs Regularization Parameter')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
