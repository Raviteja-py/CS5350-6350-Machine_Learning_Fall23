import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv("./train.csv", header=None)
test_data = pd.read_csv("./test.csv", header=None)

print("\n")
print("3a)Answer===================================================================\n")
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def log_prior(w, v):
    return -0.5 * np.sum(w**2) / v - 0.5 * np.log(2 * np.pi * v) * len(w)

from scipy.special import expit
def log_likelihood(X, y, w):
    z = np.dot(X, w)
    return np.sum(y * z - np.log(1 + expit(z)))

def log_posterior(X, y, w, v):
    return log_likelihood(X, y, w) + log_prior(w, v)

def gradient(X, y, w, v):
    z = np.dot(X, w)
    p = sigmoid(z)
    prior_grad = -w / v
    likelihood_grad = np.dot(X.T, p - y)
    return prior_grad + likelihood_grad

def sgd(X, y, v, lr_0, d, epochs):
    w = np.zeros(X.shape[1])
    log_likelihood_values = []    
    for epoch in range(epochs):
        X, y = shuffle(X, y)
        lr_t = lr_0 / (1 + (lr_0 / d) * epoch)        
        for i in range(len(X)):
            grad = gradient(X[i:i+1], y[i:i+1], w, v)
            w += lr_t * grad
        log_likelihood_values.append(log_posterior(X, y, w, v))    
    return w, log_likelihood_values

variances = [0.01, 0.1, 0.5, 1.0,3.0,5.0,10.0,100.0]
lr_0 = 0.01
d = 1.0
epochs = 100

for v in variances:
    weights, log_likelihood_values = sgd(X_train_bias, y_train, v, lr_0, d, epochs)
    train_predictions = sigmoid(np.dot(X_train_bias, weights))
    train_predictions_binary = (train_predictions > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, train_predictions_binary)
    train_error = 1-train_accuracy
    test_predictions = sigmoid(np.dot(X_test_bias, weights))
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_predictions_binary)
    test_error = 1- test_accuracy
    print(f"Variance={v}, Training Error: {train_error}, Test Error: {test_error}")

print("\n")
print("3b)Answer===================================================================\n")
def ml_sgd(X, y, lr_0, d, epochs):
    w = np.zeros(X.shape[1])
    log_likelihood_values = []
    for epoch in range(epochs):
        X, y = shuffle(X, y)
        lr_t = lr_0 / (1 + (lr_0 / d) * epoch)
        for i in range(len(X)):
            gradient_val = ml_gradient(X[i], y[i], w)
            w += lr_t * gradient_val
            log_likelihood_val = ml_log_likelihood(X[i], y[i], w)
            log_likelihood_values.append(log_likelihood_val)
    return w, log_likelihood_values

def ml_gradient(X, y, w):
    z = np.dot(X, w)
    p = sigmoid(z)
    return np.dot(X.T, p - y)

def ml_log_likelihood(X, y, w):
    z = np.dot(X, w)
    return y * z - np.log(1 + expit(z))

ml_lr_0 = 0.01
ml_d = 1.0

for v in variances:
    ml_weights, ml_log_likelihood_values = ml_sgd(X_train_bias, y_train, ml_lr_0, ml_d, epochs)
    ml_train_predictions = sigmoid(np.dot(X_train_bias, ml_weights))
    ml_train_predictions_binary = (ml_train_predictions > 0.5).astype(int)
    ml_train_accuracy = accuracy_score(y_train, ml_train_predictions_binary)
    ml_train_error = 1- ml_train_accuracy
    ml_test_predictions = sigmoid(np.dot(X_test_bias, ml_weights))
    ml_test_predictions_binary = (ml_test_predictions > 0.5).astype(int)
    ml_test_accuracy = accuracy_score(y_test, ml_test_predictions_binary)
    ml_test_error = 1- ml_test_accuracy
    print(f"Variance={v}, ML Training error: {ml_train_error} ML Test error: {ml_test_error}")
