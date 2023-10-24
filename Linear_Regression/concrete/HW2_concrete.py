""" Answer 4==================================================================="""
import random
import math
import matplotlib.pyplot as plt

# Load the data from CSV files
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            data.append([float(x) for x in values])
    return data

def predict(features, weights):
    return sum(x * w for x, w in zip(features, weights))

def batch_gradient_descent(train_data, test_data, learning_rate, tolerance=1e-6):
    num_features = len(train_data[0]) - 1
    weights = [0] * num_features
    cost_history = []
    
    def compute_cost(data, weights):
        total_cost = 0
        for example in data:
            features = example[:-1]
            target = example[-1]
            prediction = predict(features, weights)
            total_cost += (prediction - target) ** 2
        return total_cost / (2 * len(data))

    iteration = 0
    while True:
        old_weights = list(weights)
        cost = compute_cost(train_data, weights)
        cost_history.append(cost)
        
        for i in range(num_features):
            gradient = sum((predict(example[:-1], weights) - example[-1]) * example[i] for example in train_data)
            weights[i] -= learning_rate * gradient
        
        weight_difference = sum((w1 - w2) ** 2 for w1, w2 in zip(weights, old_weights))
        if weight_difference < tolerance:
            break

        iteration += 1
    
    test_cost = compute_cost(test_data, weights)
    
    return weights, cost_history, test_cost

def stochastic_gradient_descent(train_data, test_data, learning_rate, tolerance=1e-6):
    num_features = len(train_data[0]) - 1
    weights = [0] * num_features
    cost_history = []

    def compute_cost(data, weights):
        total_cost = 0
        for example in data:
            features = example[:-1]
            target = example[-1]
            prediction = predict(features, weights)
            total_cost += (prediction - target) ** 2
        return total_cost / (2 * len(data))

    iteration = 0
    while True:
        old_weights = list(weights)
        cost = compute_cost(train_data, weights)
        cost_history.append(cost)
        
        random.shuffle(train_data)
        for example in train_data:
            features = example[:-1]
            target = example[-1]
            prediction = predict(features, weights)
            
            for i in range(num_features):
                gradient = (prediction - target) * features[i]
                weights[i] -= learning_rate * gradient
        
        weight_difference = sum((w1 - w2) ** 2 for w1, w2 in zip(weights, old_weights))
        if weight_difference < tolerance:
            break

        iteration += 1

    test_cost = compute_cost(test_data, weights)

    return weights, cost_history, test_cost

train_data = load_data("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\concrete\\train.csv")
test_data = load_data("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\concrete\\test.csv")


# Set the learning rates for batch and stochastic gradient descent
batch_learning_rate = 0.01
sgd_learning_rate = 0.01  # You can adjust these values

# Run batch gradient descent
batch_weights, batch_cost_history, test_cost_batch = batch_gradient_descent(train_data, test_data, batch_learning_rate)

# Run stochastic gradient descent
sgd_weights, sgd_cost_history, test_cost_sgd = stochastic_gradient_descent(train_data, test_data, sgd_learning_rate)

# Print learned weights and test data cost
print("Batch Gradient Descent - Learned Weights:", batch_weights)
print("Test Data Cost (Batch Gradient Descent):", test_cost_batch)
print("Stochastic Gradient Descent - Learned Weights:", sgd_weights)
print("Test Data Cost (Stochastic Gradient Descent):", test_cost_sgd)

plt.figure()
plt.plot(range(len(batch_cost_history)), batch_cost_history, label='Batch GD') 
plt.plot(range(len(sgd_cost_history)), sgd_cost_history, label='SGD')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value') 
plt.legend()
plt.show()


# Run batch gradient descent
batch_weights, batch_cost_history, test_cost_batch = batch_gradient_descent(train_data, test_data, batch_learning_rate)

# Plot cost function values for batch gradient descent
plt.figure()
plt.plot(range(len(batch_cost_history)), batch_cost_history, label='Batch GD') 
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value') 
plt.legend()
plt.title('Batch Gradient Descent')
plt.show()

# Run stochastic gradient descent
sgd_weights, sgd_cost_history, test_cost_sgd = stochastic_gradient_descent(train_data, test_data, sgd_learning_rate)

# Plot cost function values for stochastic gradient descent
plt.figure()
plt.plot(range(len(sgd_cost_history)), sgd_cost_history, label='SGD')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value') 
plt.legend()
plt.title('Stochastic Gradient Descent')
plt.show()

# Print learned weights and test data cost
print("Batch Gradient Descent - Learned Weights:", batch_weights)
print("Test Data Cost (Batch Gradient Descent):", test_cost_batch)
print("Stochastic Gradient Descent - Learned Weights:", sgd_weights)
print("Test Data Cost (Stochastic Gradient Descent):", test_cost_sgd)



