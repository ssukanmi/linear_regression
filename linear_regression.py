import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_data(x_values, y_values):
    fig1 = plt.figure()
    plt.scatter(x_values, y_values, 10, 'r', 'x', label='training set')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.xticks(range(4, 25, 2))
    plt.axis([4, 25, -5, 26])
    return fig1


def compute_cost(x_values, y_values, theta_values):
    m = len(y_values)
    y_pred = x_values.dot(theta_values)
    sqr_err = (y_pred - y_values) ** 2
    return 1/(2*m) * np.sum(sqr_err)


def gradient_descent(x_values, y_values, theta_values, learning_rate, num_iters):
    m = len(y_values)
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        hyp = x_values.dot(theta_values)
        err = hyp - y_values
        theta_change = alpha * 1/m * x_values.transpose().dot(err)
        theta_values -= theta_change
        j_history[i] = compute_cost(x_values, y_values, theta_values)
    return theta_values, j_history


def feature_norm(x_values):
    x_norm = x_values
    mu = np.zeros((1, np.size(x_values, 1)))
    sigma = np.zeros((1, np.size(x_values, 1)))


data1 = pd.read_csv('ex1data1.txt', header=None).to_numpy()
X = data1[:, [0]]
y = data1[:, [1]]

plot_data(X, y)
# plt.show()

m = len(X)  # number of training examples
X = np.append(np.ones((m, 1)), X, 1)
theta = np.zeros((2, 1))  # initial fitting parameters
alpha = 0.01  # learning rate
iterations = 1500

print(compute_cost(X, y, theta))
print(compute_cost(X, y, np.array([[-1], [2]])))

theta, j_history = gradient_descent(X, y, theta, alpha, iterations)
# print(j_history)
print(f'Theta computed from gradient descent:\n{theta[0, 0]}, {theta[1, 0]}')

plt.plot(X[:, [1]], X.dot(theta), 'b', label='linear regression')
plt.legend()
# plt.show()

# predict values for population sizes of 35,000 and 70,000
prediction1 = np.array([1, 3.5]).dot(theta) * 10000
prediction2 = np.array([1, 7]).dot(theta) * 10000
print(f'For population = 35,000, we predict a profit of {prediction1}')
print(f'For population = 70,000, we predict a profit of {prediction2}')


theta0_values = np.linspace(-10, 10, 100)
theta1_values = np.linspace(-1, 4, 100)
j_vals = np.zeros((len(theta0_values), len(theta1_values)))

for i in range(len(theta0_values)):
    for j in range(len(theta1_values)):
        t = np.array([[theta0_values[i]], [theta1_values[j]]])
        j_vals[i, j] = compute_cost(X, y, t)

j_vals = j_vals.transpose()


fig2 = plt.figure(figsize=(14, 6))
ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(theta0_values, theta1_values, j_vals)
ax2 = fig2.add_subplot(1, 2, 2)
CS = ax2.contour(theta0_values, theta1_values, j_vals, np.logspace(-2, 3, 20))
ax2.plot(theta[0, 0], theta[1, 0], marker='x', color='r')
ax2.set_xlabel('theta0')
ax2.set_ylabel('theta1')
plt.show()

data = pd.read_csv('ex1data2.txt', header=None).to_numpy()
X = data[:, [0, 1]]
y = data[:, [2]]

# print(np.mean(X, axis=0))
# print(np.std(X, axis=0))
