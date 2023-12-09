import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# Load the dataset
train_data = pd.read_csv("./train.csv", header=None)
test_data = pd.read_csv("./test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

input_size = X_train.shape[1]  
hidden_size = 5  
output_size = 1

np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, output_size)

def backpropagation(x, y, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate=0.01):
    hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    weights_hidden_output_gradient = hidden_layer_output.T.dot(d_predicted_output)
    weights_input_hidden_gradient = x.reshape(-1, 1).dot(d_hidden_layer.reshape(1, -1))
    bias_output_gradient = d_predicted_output
    bias_hidden_gradient = d_hidden_layer
    weights_input_hidden += learning_rate * weights_input_hidden_gradient
    weights_hidden_output += learning_rate * weights_hidden_output_gradient
    bias_hidden += learning_rate * bias_hidden_gradient
    bias_output += learning_rate * bias_output_gradient
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def forward_pass(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)    
    return output

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = backpropagation(
    X_train[0], y_train[0],
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

print("2a)Answer===================================================================\n")
print("Updated weights from input to hidden layer:\n", weights_input_hidden)
print("Updated weights from hidden to output layer:\n", weights_hidden_output)
print("Updated bias for hidden layer:", bias_hidden)
print("Updated bias for output layer:", bias_output)

gamma_0 = 0.01
d = 1.0
epochs = 100
widths = [5, 10, 25, 50, 100]

print("\n")
print("2b)Answer===================================================================\n")
for width in widths:
    weights_input_hidden = np.random.randn(X_train.shape[1], width)
    weights_hidden_output = np.random.randn(width, 1)
    bias_hidden = np.random.randn(1, width)
    bias_output = np.random.randn(1, 1)

    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        learning_rate = gamma_0 / (1 + gamma_0 * epoch / d)
        for i in range(len(X_train)):
            weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = backpropagation(
                X_train[i].reshape(1, -1), y_train[i], 
                weights_input_hidden, weights_hidden_output,
                bias_hidden, bias_output,
                learning_rate)

    train_predictions = forward_pass(X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    test_predictions = forward_pass(X_test, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

    train_error = mean_squared_error(y_train, train_predictions)
    test_error = mean_squared_error(y_test, test_predictions)

    print(f"Width: {width}, Training Error: {train_error}, Test Error: {test_error}")

print("\n")
print("2c)Answer===================================================================\n")
for width in widths:
    weights_input_hidden = np.zeros((X_train.shape[1], width))
    weights_hidden_output = np.zeros((width, 1))
    bias_hidden = np.zeros((1, width))
    bias_output = np.zeros((1, 1))

    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        learning_rate = gamma_0 / (1 + gamma_0 * epoch / d)

        for i in range(len(X_train)):
            weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = backpropagation(
                X_train[i].reshape(1, -1), y_train[i],
                weights_input_hidden, weights_hidden_output,
                bias_hidden, bias_output,
                learning_rate)

    train_predictions = forward_pass(X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    test_predictions = forward_pass(X_test, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

    train_error = mean_squared_error(y_train, train_predictions)
    test_error = mean_squared_error(y_test, test_predictions)

    print(f"Width: {width}, Training Error: {train_error}, Test Error: {test_error}")

print("\n")
print("2e)Answer===================================================================\n")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = activation()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def train_neural_network(X_train, y_train, X_test, y_test, depth, width, activation):
    input_size = X_train.shape[1]
    output_size = 1
    model = NeuralNetwork(input_size, width, output_size, activation)
    if activation == nn.Tanh:
        model.apply(xavier_init)
    elif activation == nn.ReLU:
        model.apply(he_init)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    train_predictions = model(X_train_tensor).detach().numpy()
    test_predictions = model(X_test_tensor).detach().numpy()
    train_error = mean_squared_error(y_train, train_predictions)
    test_error = mean_squared_error(y_test, test_predictions)
    return train_error, test_error

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]

for activation in [nn.Tanh, nn.ReLU]:
    for depth in depths:
        for width in widths:
            train_error, test_error = train_neural_network(X_train, y_train, X_test, y_test, depth, width, activation)
            print(f"Activation: {activation.__name__}, Depth: {depth}, Width: {width}, Training Error: {train_error}, Test Error: {test_error}")
