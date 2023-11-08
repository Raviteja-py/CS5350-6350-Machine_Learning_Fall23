import pandas as pd
import numpy as np

# function to shuffle data
def shuffle(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# prediction error
def calculate_error(w, X, y):
    predictions = np.sign(np.dot(X, w))
    return np.mean(predictions != y)

# Standard Perceptron
def standard_perceptron(X_train, y_train, X_test, y_test, T):
    # Initialize weight vector and bias
    w = np.zeros(X_train.shape[1])
    b = 0

    for t in range(T):
        # Shuffle the training data
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(len(X_train)):
            x_i = X_train[i]
            y_i = y_train[i]
            prediction = np.dot(w, x_i) + b

            if prediction * y_i <= 0:
                w = w + y_i * x_i
                b = b + y_i

        test_error = calculate_error(w, X_test, y_test)
        print(f"Epoch {t + 1}, Test Error: {test_error:.4f}")

    return w, b

# Voted Perceptron
def voted_perceptron(X_train, y_train, X_test, y_test, T):
    w_list = []
    c_list = []

    # Initialize weight vector and bias
    w = np.zeros(X_train.shape[1])
    b = 0

    for t in range(T):
        X_train, y_train = shuffle(X_train, y_train)
        num_updates = 0

        for i in range(len(X_train)):
            x_i = X_train[i]
            y_i = y_train[i]
            prediction = np.dot(w, x_i) + b

            if prediction * y_i <= 0:
                w_list.append(w.copy())
                c_list.append(num_updates)
                w = w + y_i * x_i
                b = b + y_i
                num_updates = 1
            else:
                num_updates += 1

    # Calculate final weighted average weights
    final_weights = np.zeros(X_train.shape[1])
    for i, w in enumerate(w_list):
        final_weights += c_list[i] * w

    # Calculate test error
    test_error = calculate_error(final_weights, X_test, y_test)
    return final_weights, test_error

# Average Perceptron
def average_perceptron(X_train, y_train, X_test, y_test, T):
    w_sum = np.zeros(X_train.shape[1])
    b_sum = 0
    num_updates = 0

    for t in range(T):
        X_train, y_train = shuffle(X_train, y_train)

        for i in range(len(X_train)):
            x_i = X_train[i]
            y_i = y_train[i]
            prediction = np.dot(w_sum, x_i) + b_sum

            if prediction * y_i <= 0:
                w_sum = w_sum + y_i * x_i
                b_sum = b_sum + y_i
                num_updates += 1

    # Calculate the average weights and bias
    avg_weights = w_sum / num_updates
    avg_bias = b_sum / num_updates

    # Calculate test error
    test_error = calculate_error(avg_weights, X_test, y_test)
    return avg_weights, avg_bias, test_error

# Read train dataset
df_train = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW3\\bank\\train.csv")
df_train.head()
df_train.describe()
# sns.pairplot(df_train)


# Read test dataset
df_test = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW3\\bank\\test.csv")
df_test.head()
df_test.describe()
# sns.pairplot(df_test)

# Extract features and labels for training data
X_train = df_train.iloc[:, :-1].values  # Features
y_train = df_train.iloc[:, -1].values   # Labels

# Extract features and labels for test data
X_test = df_test.iloc[:, :-1].values  # Features
y_test = df_test.iloc[:, -1].values   # Labels

# maximum number of epochs
T = 10

# Standard Perceptron
print("Standard Perceptron")
w_standard, b_standard = standard_perceptron(X_train, y_train, X_test, y_test, T)
test_error_standard = calculate_error(w_standard, X_test, y_test)
print(f"Learned weight vector: {w_standard}")
print(f"Average prediction error on test data: {test_error_standard:.4f}\n")

# Voted Perceptron
print("Voted Perceptron")
w_voted, test_error_voted = voted_perceptron(X_train, y_train, X_test, y_test, T)
print(f"Learned weight vector: {w_voted}")
print(f"Average test error: {test_error_voted:.4f}\n")

# Average Perceptron
print("Average Perceptron")
w_avg, b_avg, test_error_avg = average_perceptron(X_train, y_train, X_test, y_test, T)
print(f"Learned weight vector: {w_avg}")
print(f"Learned bias: {b_avg}")
print(f"Average prediction error on test data: {test_error_avg:.4f}\n")
