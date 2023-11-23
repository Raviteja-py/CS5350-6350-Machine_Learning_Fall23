print("Results for 2a)=======================================================================================\n")

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

# learning rate schedule
def learning_rate_schedule_1(t, gamma_0, a):
    return gamma_0 / (1 + (gamma_0 / a) * t)

# calculate the error rate
def calculate_error(w, b, data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    predictions = np.sign(np.dot(X, w) + b)
    errors = predictions != y
    return np.mean(errors)

# SVM SGD implementation using learning rate schedule 1
def svm_sgd_schedule_1(train_data, test_data, C, gamma_0, a, max_epochs):
    # weights and bias
    w = np.zeros(train_data.shape[1] - 1)
    b = 0.0
    training_errors = []
    test_errors = []

    for epoch in range(max_epochs):
        train_data = shuffle(train_data)
        for i in range(len(train_data)):
            t = epoch * len(train_data) + i  
            x_i = train_data.iloc[i, :-1].values
            y_i = train_data.iloc[i, -1]
            learning_rate = learning_rate_schedule_1(t, gamma_0, a)

            if y_i * (np.dot(w, x_i) + b) < 1:
                w = w - learning_rate * (w - C * y_i * x_i)
                b = b - learning_rate * (-C * y_i)
            else:
                w = w - learning_rate * w
        # errors after each epoch
        train_error = calculate_error(w, b, train_data)
        test_error = calculate_error(w, b, test_data)
        training_errors.append(train_error)
        test_errors.append(test_error)
    return w, b, training_errors, test_errors

# Load the dataset
train_data = pd.read_csv("C:\\Users\\ravit\\Desktop\\Fall 2023\\ML\\HW4\\bank_note\\train.csv",header=None)
test_data = pd.read_csv("C:\\Users\\ravit\\Desktop\\Fall 2023\\ML\\HW4\\bank_note\\test.csv",header=None)

# Convert labels to {1, -1}
train_data.iloc[:, -1] = train_data.iloc[:, -1].replace(0, -1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace(0, -1)

# Hyperparameters
C1_values = [100/873, 500/873, 700/873]
gamma_0 = 0.01  
a = 10  
max_epochs = 100

# SVM SGD for each value of C using learning rate schedule 1
for C1 in C1_values:
    w, b, training_errors, test_errors = svm_sgd_schedule_1(train_data, test_data, C1, gamma_0, a, max_epochs)
    print(f'For C={C1}:')
    print(f'Final training error: {training_errors[-1]}')
    print(f'Final test error: {test_errors[-1]}')
    print(f'Final model parameters: w={w}, b={b}\n')

 
print("Results for 2b)=======================================================================================\n")
# learning rate schedule 2
def learning_rate_schedule_2(t, gamma_0):
    return gamma_0 / (1 + t)

# SVM SGD implementation using learning rate schedule 2
def svm_sgd_schedule_2(train_data, test_data, C, gamma_0, max_epochs):
    w = np.zeros(train_data.shape[1] - 1)
    b = 0.0
    training_errors = []
    test_errors = []

    for epoch in range(max_epochs):
        train_data = shuffle(train_data)
        for i in range(len(train_data)):
            t = epoch * len(train_data) + i
            x_i = train_data.iloc[i, :-1].values
            y_i = train_data.iloc[i, -1]
            learning_rate = learning_rate_schedule_2(t, gamma_0)
 
            if y_i * (np.dot(w, x_i) + b) < 1:
                w = w - learning_rate * (w - C * y_i * x_i)
                b = b - learning_rate * (-C * y_i)
            else:
                w = w - learning_rate * w
        # Calculate errors after each epoch
        train_error = calculate_error(w, b, train_data)
        test_error = calculate_error(w, b, test_data)
        training_errors.append(train_error)
        test_errors.append(test_error)
    return w, b, training_errors, test_errors

# Hyperparameters
C2_values = [100/873, 500/873, 700/873]
gamma_0 = 0.01
max_epochs = 100


# Run SVM SGD for each value of C using learning rate schedule 2
for C2 in C2_values:
    w, b, training_errors, test_errors = svm_sgd_schedule_2(train_data, test_data, C2, gamma_0, max_epochs)
    print(f'For C={C2}:')
    print(f'Final training error: {training_errors[-1]}')
    print(f'Final test error: {test_errors[-1]}')
    print(f'Final model parameters: w={w}, b={b}\n')


print("Results for 2c)====================================================================================\n")

def compare_schedules(training_errors_1, test_errors_1, training_errors_2, test_errors_2):
    print("Comparison of schedules learning rates :\n")
    for C in C1_values:  
        print(f"For C={C}:")
        print(f"Difference in training error: {abs(training_errors_1[C][-1] - training_errors_2[C][-1])}")
        print(f"Difference in test error: {abs(test_errors_1[C][-1] - test_errors_2[C][-1])}\n")

training_errors_1 = {}
test_errors_1 = {}
training_errors_2 = {}
test_errors_2 = {}

for C in C1_values:
    _, _, training_errors, test_errors = svm_sgd_schedule_1(train_data, test_data, C, gamma_0, a, max_epochs)
    training_errors_1[C] = training_errors
    test_errors_1[C] = test_errors

for C in C2_values:
    _, _, training_errors, test_errors = svm_sgd_schedule_2(train_data, test_data, C, gamma_0, max_epochs)
    training_errors_2[C] = training_errors
    test_errors_2[C] = test_errors

compare_schedules(training_errors_1, test_errors_1, training_errors_2, test_errors_2)

for C in C1_values:
    w_1, b_1, _, _ = svm_sgd_schedule_1(train_data, test_data, C, gamma_0, a, max_epochs)
    w_2, b_2, _, _ = svm_sgd_schedule_2(train_data, test_data, C, gamma_0, max_epochs)
    w_diff = np.linalg.norm(w_1 - w_2)
    b_diff = abs(b_1 - b_2)
    
    print(f"Differences for C={C}:")
    print(f"Weight vector difference (L2 norm): {w_diff}")
    print(f"Bias difference: {b_diff}")
    print("\n")


print("Results for 3a)====================================================================================\n")
train_data.iloc[:, -1] = train_data.iloc[:, -1].replace({0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace({0: -1})

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def objective_function(alpha, y, K):
    half_quad_term = np.dot(alpha * y, np.dot(K, alpha * y))
    return -np.sum(alpha) + 0.5 * half_quad_term

def zero_constraint(alpha, y):
    return np.dot(alpha, y)

# Train the dual SVM
def train_dual_svm(X, y, C):
    n_samples, n_features = X.shape
    K = linear_kernel(X, X)
    initial_alphas = np.zeros(n_samples)
    bounds = [(0, C) for _ in range(n_samples)]
    constraint = {'type': 'eq', 'fun': zero_constraint, 'args': (y,)}
    result = minimize(fun=objective_function, x0=initial_alphas, args=(y, K),
                      method='SLSQP', bounds=bounds, constraints=[constraint])

    alphas = result.x
    sv_mask = alphas > 1e-5
    support_vectors = X[sv_mask]
    support_vector_labels = y[sv_mask]
    support_vector_alphas = alphas[sv_mask]
    w = np.sum(support_vectors.T * support_vector_labels * support_vector_alphas, axis=1)
    b = np.mean(support_vector_labels - np.dot(support_vectors, w))
    return w, b, alphas, sv_mask

# Values for C
C_values = [100/873, 500/873, 700/873]

# Training and comparing
for C in C_values:
    w, b, alphas, support_vectors_mask = train_dual_svm(X_train, y_train, C)
    print(f"Trained dual SVM with C={C}")
    print(f"Weights: {w}")
    print(f"Bias: {b}")
    print(f"Support Vectors: {np.sum(support_vectors_mask)}\n")

print("\n")
print("Results for 3b)=======================================================================================\n")
def gaussian_kernel_matrix(X, sigma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-np.sum((X[i, :] - X[j, :]) ** 2) / (2 * sigma ** 2))
    return K

def train_dual_svm_gaussian(X, y, C, sigma):
    K = gaussian_kernel_matrix(X, sigma)
   
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha * y) * y) - np.sum(alpha)
    
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y), 'jac': lambda alpha: y}
    bounds = [(0, C) for _ in range(X.shape[0])]

    result = minimize(fun=objective,
                      x0=np.zeros(X.shape[0]),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    alphas = result.x
    sv = (alphas > 1e-5)
    b = np.mean(y[sv] - np.dot(K[sv], alphas * y))
    return alphas, b, sv

# SVM prediction function
def svm_predict(X, X_sv, y_sv, alphas_sv, b, gamma):
    K = np.zeros((X.shape[0], X_sv.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X_sv.shape[0]):
            K[i, j] = np.exp(-np.sum((X[i, :] - X_sv[j, :]) ** 2) / (2 * gamma ** 2))
    predictions = np.dot(K, alphas_sv * y_sv) + b
    return np.sign(predictions)

X_train_np = X_train
y_train_np = y_train
X_test_np = X_test
y_test_np = y_test

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873]

# Training and prediction loop
for C in C_values:
    for gamma in gamma_values:
        alphas, b, sv = train_dual_svm_gaussian(X_train_np, y_train_np, C, gamma)
        X_sv = X_train_np[sv]
        y_sv = y_train_np[sv]
        alphas_sv = alphas[sv]
        y_train_pred = svm_predict(X_train_np, X_sv, y_sv, alphas_sv, b, gamma)
        y_test_pred = svm_predict(X_test_np, X_sv, y_sv, alphas_sv, b, gamma)
        
        # Calculate errors
        train_error = np.mean(y_train_pred != y_train_np)
        test_error = np.mean(y_test_pred != y_test_np)
        print(f"Results for C={C} and gamma={gamma}:")
        print(f"Training error: {train_error}")
        print(f"Test error: {test_error}")

    
print("\n")        
print("Results for 3c)=======================================================================================\n")
C_value = 500 / 873
gamma_values = [0.01, 0.1, 0.5, 1, 5, 10]  
support_vector_indices = {}

for gamma in gamma_values:
    alphas, _, sv = train_dual_svm_gaussian(X_train_np, y_train_np, C_value, gamma)
    support_vector_indices[gamma] = np.where(sv)[0]

for i in range(len(gamma_values) - 1):
    gamma1 = gamma_values[i]
    gamma2 = gamma_values[i + 1]
    overlap = np.intersect1d(support_vector_indices[gamma1], support_vector_indices[gamma2])
    print(f"Number of overlapping support vectors between gamma={gamma1} and gamma={gamma2}: {len(overlap)}")

print("\n")
print("Results for 3d)=======================================================================================\n")

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def train_kernel_perceptron(X, y, gamma, epochs):
    n_samples, n_features = X.shape
    alphas = np.zeros(n_samples)

    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)

    for epoch in range(epochs):
        for i in range(n_samples):
            if np.sign(np.sum(K[:, i] * alphas * y)) != y[i]:
                alphas[i] += 1
    return alphas

def kernel_perceptron_predict(X, X_train, y_train, alphas, gamma):
    y_pred = []
    for x in X:
        prediction = np.sign(np.sum([alphas[i] * y_train[i] * gaussian_kernel(x, X_train[i], gamma) for i in range(len(X_train))]))
        y_pred.append(prediction)
    return np.array(y_pred)

def calculate_errors(X_train, y_train, X_test, y_test, gamma_values):
    errors = {}
    for gamma in gamma_values:
        alphas = train_kernel_perceptron(X_train, y_train, gamma, epochs=10)
        y_train_pred = kernel_perceptron_predict(X_train, X_train, y_train, alphas, gamma)
        y_test_pred = kernel_perceptron_predict(X_test, X_train, y_train, alphas, gamma)
        train_error = np.mean(y_train_pred != y_train)
        test_error = np.mean(y_test_pred != y_test)
        errors[gamma] = (train_error, test_error)
    return errors

gamma_values = [0.1, 0.5, 1, 5, 100]
errors = calculate_errors(X_train_np, y_train_np, X_test_np, y_test_np, gamma_values)

for gamma in gamma_values:
    train_error, test_error = errors[gamma]
    print(f"Gamma: {gamma}, Training error: {train_error}, Test error: {test_error}")


