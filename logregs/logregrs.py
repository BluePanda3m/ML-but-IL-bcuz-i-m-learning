import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 + y) * np.log(1 - p))


X = np.array([[0.5], [1.0], [1.5], [3.0], [3.5], [4.0]])
y = np.array([0, 0, 0, 1, 1, 1])

w = np.zeros(1)
b = 0.0
lr = 0.5

for steps in range(1000):
    p = sigmoid(X @ w + b)
    grad_w = X.T @ (p - y) / len(y)
    grad_b = np.mean(p - y)
    w -= lr * grad_w
    b -= lr * grad_b
    if steps % 200 == 0:
        print(f"steps {steps:4d} loss {bce(y, p): .4f}")

print("w =", w, "b =", b)
