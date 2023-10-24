"""Answer for 2)a """

from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load train and test data sets
train_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\bank-7\\train.csv")
test_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\bank-7\\test.csv")


# Print the given labels
attribute_names = train_data.columns
print("Labels:", attribute_names)

# Function to calculate entropy
def entropy(y):
    if len(y) == 0:
        return 0
    p = np.array(list(Counter(y).values())) / len(y)
    return -np.sum(p * np.log2(p))

# Function to calculate information gain
def information_gain(y, subsets):
    total_entropy = entropy(y)
    weighted_entropy = sum((len(subset) / len(y)) * entropy(subset) for subset in subsets)
    return total_entropy - weighted_entropy

def decision_stump(data, weights, attributes, target_attribute):
    best_attribute = None
    best_threshold = None
    best_score = None

    for attribute in attributes:
        if attribute != target_attribute:
            unique_values = data[attribute].unique()
            unique_values = np.append(unique_values, "unknown")  # Consider "unknown" as a valid attribute value

            for value in unique_values:
                value_str = str(value)  # Convert the value to a string
                predicted = np.where(data[attribute].astype(str) <= value_str, True, False)
                subsets = [data[target_attribute][predicted], data[target_attribute][~predicted]]

                score = information_gain(data[target_attribute], subsets)

                if best_score is None or score > best_score:
                    best_score = score
                    best_attribute = attribute
                    best_threshold = value

    return best_attribute, best_threshold


# AdaBoost training
def adaboost_train(data, attributes, target, T):
    weights = np.ones(len(data)) / len(data)
    stumps = []
    train_errors = []
    test_errors = []
    
    for t in range(T):
        # Learn decision stump
        chosen_attr, threshold = decision_stump(data, weights, attributes, target)
        
        # Make predictions
        predictions = np.where(data[target] == 'yes', 1, -1)
        stump_preds = predictions != (data[chosen_attr] <= threshold)
        
        # Calculate error
        error = np.sum(weights[stump_preds])
        
        # Update weights
        alpha = 0.5 * np.log((1 - error) / error)
        weights *= np.exp(-alpha * stump_preds)
        weights /= np.sum(weights)
        
        # Evaluate train and test error
        train_preds = adaboost_predict(train_data, stumps)
        test_preds = adaboost_predict(test_data, stumps)
        train_correct = np.sum(train_preds == train_data[target])
        test_correct = np.sum(test_preds == test_data[target])
        train_error = 1 - train_correct / len(train_data)
        test_error = 1 - test_correct / len(test_data)
        
        # Store errors
        train_errors.append(train_error) 
        test_errors.append(test_error)
        
        # Store stump
        stumps.append((chosen_attr, threshold, alpha))
        
    return stumps, train_errors, test_errors


# AdaBoost Prediction
def adaboost_predict(data, stumps):
    predictions = np.zeros(len(data))
    for attr, thresh, alpha in stumps:
        predictions += alpha * (data[attr] <= thresh)
    return np.where(predictions >= 0, 'yes', 'no')


# Dataset column names 
attributes = list(attribute_names)
target = 'y'

# Train
T = 500
stumps, train_errors, test_errors = adaboost_train(train_data, attributes, target, T)


# Print final errors
train_preds = adaboost_predict(train_data, stumps)
test_preds = adaboost_predict(test_data, stumps)

train_error = np.mean(train_preds != train_data[target]) 
test_error = np.mean(test_preds != test_data[target])

print("Final train error:", train_error)
print("Final test error:", test_error)
print("Stumps:",stumps)
print("Train_Errors:",train_errors)
print("Test_errors:",test_errors)

# Plot train and test errors
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(T), train_errors, label='Train')
plt.plot(range(T), test_errors, label='Test')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations')

# Plot train and test errors for each decision stump
train_stump_errors = []
test_stump_errors = []

for stump in stumps:
    train_preds = adaboost_predict(train_data, [stump])
    test_preds = adaboost_predict(test_data, [stump])
    train_error = np.mean(train_preds != train_data[target]) 
    test_error = np.mean(test_preds != test_data[target])
    train_stump_errors.append(train_error)
    test_stump_errors.append(test_error)

plt.subplot(1, 2, 2)
plt.plot(range(T), train_stump_errors, label='Train')
plt.plot(range(T), test_stump_errors, label='Test')
plt.legend()
plt.xlabel('Stump')
plt.ylabel('Error')
plt.title('Error vs Stump')

plt.tight_layout()
plt.show()


""" Answer for 2)b and 2)c ===================================================="""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load train and test data sets
train_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\bank-7\\train.csv")
test_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\bank-7\\test.csv")

# Define a function to calculate entropy
def entropy(counts):
    total = sum(counts)
    entropy_value = 0
    for element in counts:
        p = (element / total)
        if p != 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

# Define a function to calculate information gain
def information_gain(X, Y, attribute):
    _, counts = np.unique(Y, return_counts=True)
    entropy_attribute = entropy(counts)
    entropy_parent = 0
    distinct_attr_values = list(set(X[:, attribute]))
    for val in distinct_attr_values:
        indices = np.where(X[:, attribute] == val)[0]
        _, counts = np.unique(Y[indices], return_counts=True)
        entr = entropy(counts)
        entropy_parent += (len(indices) / len(Y)) * entr
    info_gain = entropy_attribute - entropy_parent
    return info_gain, entropy_attribute, entropy_parent

# Define a function to build a decision tree
def ID3(X, Y, attribute_list, current_depth=0):
    if current_depth >= max_depth or len(attribute_list) == 0 or len(np.unique(Y)) == 1:
        vals, counts = np.unique(Y, return_counts=True)
        return vals[np.argmax(counts)]

    max_info_gain = -1
    max_attribute = None
    for attribute in attribute_list:
        info_gain, entropy_attribute, entropy_parent = information_gain(X, Y, attribute)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_attribute = attribute

    vals, counts = np.unique(Y, return_counts=True)
    root_label = vals[np.argmax(counts)]

    attribute_values = np.unique(X[:, max_attribute])
    new_attribute_list = [attr for attr in attribute_list if attr != max_attribute]

    tree = {
        "attribute": max_attribute,
        "label": root_label,
        "trees": {}
    }

    for value in attribute_values:
        indices = np.where(X[:, max_attribute] == value)[0]
        if len(indices) == 0:
            tree["trees"][value] = root_label
        else:
            tree["trees"][value] = ID3(X[indices], Y[indices], new_attribute_list, current_depth + 1)

    return tree


# Define a function to predict using a decision tree
def predict(tree, x):
    if isinstance(tree, dict):
        attribute = tree["attribute"]
        value = x[attribute]
        if value in tree["trees"]:
            return predict(tree["trees"][value], x)
        else:
            return tree["label"]
    else:
        return tree

# Define a function to fit a decision tree
def fit_decision_tree(X, Y):
    attribute_list = list(range(X.shape[1]))  # Assume attributes are indexed
    decision_tree = ID3(X, Y, attribute_list)
    return decision_tree

# Define a function to perform bagged trees
def bagged_trees(X, Y, num_trees):
    bagged_tree_predictions = []

    for _ in range(num_trees):
        sample_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_bootstrap, Y_bootstrap = X[sample_indices], Y[sample_indices]
        decision_tree = fit_decision_tree(X_bootstrap, Y_bootstrap)
        predictions = np.array([predict(decision_tree, x) for x in X])
        bagged_tree_predictions.append(predictions)

    return np.mean(bagged_tree_predictions, axis=0)

# Define a function to compute bias and variance
def compute_bias_variance(predictions, ground_truth):
    # Compute bias (average prediction - ground-truth label)
    bias = np.mean(predictions) - ground_truth

    # Compute variance
    variance = np.var(predictions)

    return bias, variance


X_train = train_data.drop('y', axis=1).values
y_train = train_data['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

X_test = test_data.drop('y', axis=1).values
y_test = test_data['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

max_depth = np.inf  # Set your desired max depth
num_trees = 500  # Set the number of trees

# 2b) Vary the number of trees from 1 to 500 and report training and test errors
num_trees_range = range(1, 501)
train_errors_bagging = []
test_errors_bagging = []

for num_trees in num_trees_range:
    bagged_predictions = bagged_trees(X_train, y_train, num_trees)
    train_error = 1 - accuracy_score(y_train, np.sign(bagged_predictions))
    test_predictions = bagged_trees(X_test, y_train, num_trees)
    test_error = 1 - accuracy_score(y_test, np.sign(test_predictions))
    train_errors_bagging.append(train_error)
    test_errors_bagging.append(test_error)

# Plotting the errors
plt.figure(figsize=(10, 6))
plt.plot(num_trees_range, train_errors_bagging, label='Train Error (Bagging)')
plt.plot(num_trees_range, test_errors_bagging, label='Test Error (Bagging)')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Trees (Bagging)')
plt.legend()
plt.show()

# 2c) Bias and Variance calculation
num_iterations = 100
num_bagged_trees = 500
single_tree_biases = []
single_tree_variances = []
bagged_tree_biases = []
bagged_tree_variances = []

for _ in range(num_iterations):
    # Step 1: Sample 1,000 examples uniformly without replacement from the training dataset
    n_samples = X_train.shape[0]
    sample_indices = np.random.choice(n_samples, size=1000, replace=False)
    sampled_X_train, sampled_y_train = X_train[sample_indices], y_train[sample_indices]

    # Step 2: Run the bagged trees learning algorithm based on the 1,000 training examples and learn 500 trees
    bagged_predictions = bagged_trees(sampled_X_train, sampled_y_train, num_bagged_trees)

    # Step 3: Compute bias and variance for single trees and bagged trees
    # Single trees
    single_tree_predictions = np.array([bagged_trees(X_test, y_train, 1) for _ in range(num_bagged_trees)])
    avg_single_tree_predictions = np.mean(single_tree_predictions, axis=0)
    single_tree_bias, single_tree_variance = compute_bias_variance(avg_single_tree_predictions, y_test)
    single_tree_biases.append(single_tree_bias)
    single_tree_variances.append(single_tree_variance)

    # Bagged trees
    bagged_tree_bias, bagged_tree_variance = compute_bias_variance(bagged_predictions, y_test)
    bagged_tree_biases.append(bagged_tree_bias)
    bagged_tree_variances.append(bagged_tree_variance)

    # Calculate average bias, variance, and general squared error
    single_tree_bias = single_tree_biases
    single_tree_variance = single_tree_variances
    bagged_tree_bias = bagged_tree_biases
    bagged_tree_variance = bagged_tree_variances

    # Print the results
    print("Bias for single decision tree:", single_tree_bias)
    print("Variance for single decision tree:", single_tree_variance)
    print("Bias for bagged trees:", bagged_tree_bias)
    print("Variance for bagged trees:", bagged_tree_variance)

""" Answer for 2d """
# Function to generate a bootstrap sample
def generate_bootstrap_sample(X, Y):
    n_samples = X.shape[0]
    sample_indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[sample_indices]
    Y_bootstrap = Y[sample_indices]
    return X_bootstrap, Y_bootstrap

# Modify the random_forest function for 2d
def random_forest(X, Y, num_trees, feature_subset_size, max_depth):
    forest = []
    for _ in range(num_trees):
        X_bootstrap, Y_bootstrap = generate_bootstrap_sample(X, Y)
        attributes = np.random.choice(X.shape[1], feature_subset_size, replace=False)
        tree = fit_decision_tree(X_bootstrap, Y_bootstrap, attributes, max_depth)
        forest.append(tree)
    return forest

def predict_forest(forest, X):
    num_samples = X.shape[0]
    num_classes = 2  # Assuming binary classification

    predictions = np.zeros((num_samples, num_classes), dtype=int)

    for tree in forest:
        tree_predictions = [predict(tree, x) for x in X]
        for i in range(num_samples):
            predictions[i, int(tree_predictions[i])] += 1

    final_predictions = np.argmax(predictions, axis=1)

    return final_predictions



# Modify the fit_decision_tree function to accept attribute list
def fit_decision_tree(X, Y, attribute_list, max_depth):
    decision_tree = ID3(X, Y, attribute_list, max_depth)
    return decision_tree


X = X_train
Y = y_train
X_test = X_test  
Y_test = y_test  

# Define parameters
num_trees_range = range(1,501)  # Vary the number of trees
feature_subset_sizes = [2, 4, 6]  # Vary the feature subset sizes
max_depth = 10

# Lists to store results
train_errors = []
test_errors = []

# Iterate through different numbers of trees and feature subset sizes
for feature_subset_size in feature_subset_sizes:
    train_errors_rf = []
    test_errors_rf = []

    for num_trees in num_trees_range:
        forest = random_forest(X, Y, num_trees, max_depth, feature_subset_size)
        train_predictions_rf = predict_forest(forest, X)
        test_predictions_rf = predict_forest(forest, X_test)
        
        train_error_rf = 1 - accuracy_score(Y, train_predictions_rf)
        test_error_rf = 1 - accuracy_score(Y_test, test_predictions_rf)
        
        train_errors_rf.append(train_error_rf)
        test_errors_rf.append(test_error_rf)
    
    train_errors.append(train_errors_rf)
    test_errors.append(test_errors_rf)

    # Print and plot the results
    print(f"Results for max_features = {feature_subset_size}:")
    print(f"Number of Trees: {num_trees_range}")
    print(f"Train Errors: {train_errors_rf}")
    print(f"Test Errors: {test_errors_rf}")

    plt.figure(figsize=(10, 6))
    plt.plot(num_trees_range, train_errors_rf, label=f'Train Error (RF) - {feature_subset_size} features')
    plt.plot(num_trees_range, test_errors_rf, label=f'Test Error (RF) - {feature_subset_size} features')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title(f'Training and Test Errors vs. Number of Trees (Random Forest - {feature_subset_size} features)')
    plt.legend()
    plt.show()

    
"""Answer for 3 ==============================================================="""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define a function to calculate entropy
def entropy(counts):
    total = sum(counts)
    entropy_value = 0
    for element in counts:
        p = (element / total)
        if p != 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

# Define a function to calculate information gain
def information_gain(X, Y, attribute):
    _, counts = np.unique(Y, return_counts=True)
    entropy_attribute = entropy(counts)
    entropy_parent = 0
    distinct_attr_values = list(set(X[:, attribute]))
    for val in distinct_attr_values:
        indices = np.where(X[:, attribute] == val)[0]
        _, counts = np.unique(Y[indices], return_counts=True)
        entr = entropy(counts)
        entropy_parent += (len(indices) / len(Y)) * entr
    info_gain = entropy_attribute - entropy_parent
    return info_gain, entropy_attribute, entropy_parent

# Define a function to build a decision tree
def ID3(X, Y, attribute_list, current_depth=0):
    if current_depth >= max_depth or len(attribute_list) == 0 or len(np.unique(Y)) == 1:
        vals, counts = np.unique(Y, return_counts=True)
        return vals[np.argmax(counts)]

    max_info_gain = -1
    max_attribute = None
    for attribute in attribute_list:
        info_gain, entropy_attribute, entropy_parent = information_gain(X, Y, attribute)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_attribute = attribute

    vals, counts = np.unique(Y, return_counts=True)
    root_label = vals[np.argmax(counts)]

    attribute_values = np.unique(X[:, max_attribute])
    new_attribute_list = [attr for attr in attribute_list if attr != max_attribute]

    tree = {
        "attribute": max_attribute,
        "label": root_label,
        "trees": {}
    }

    for value in attribute_values:
        indices = np.where(X[:, max_attribute] == value)[0]
        if len(indices) == 0:
            tree["trees"][value] = root_label
        else:
            tree["trees"][value] = ID3(X[indices], Y[indices], new_attribute_list, current_depth + 1)

    return tree


# Define a function to predict using a decision tree
def predict(tree, x):
    if isinstance(tree, dict):
        attribute = tree["attribute"]
        value = x[attribute]
        if value in tree["trees"]:
            return predict(tree["trees"][value], x)
        else:
            return tree["label"]
    else:
        return tree

# Define a function to fit a decision tree
def fit_decision_tree(X, Y):
    attribute_list = list(range(X.shape[1]))  # Assume attributes are indexed
    decision_tree = ID3(X, Y, attribute_list)
    return decision_tree

# Define a function to perform bagged trees
def bagged_trees(X, Y, num_trees):
    bagged_tree_predictions = []

    for _ in range(num_trees):
        sample_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_bootstrap, Y_bootstrap = X[sample_indices], Y[sample_indices]
        decision_tree = fit_decision_tree(X_bootstrap, Y_bootstrap)
        predictions = np.array([predict(decision_tree, x) for x in X])
        bagged_tree_predictions.append(predictions)

    return np.mean(bagged_tree_predictions, axis=0)

# Modify the random_forest function for 2D
def random_forest(X, Y, num_trees, feature_subset_size, max_depth):
    forest = []
    for _ in range(num_trees):
        X_bootstrap, Y_bootstrap = generate_bootstrap_sample(X, Y)
        attributes = np.random.choice(X.shape[1], feature_subset_size, replace=False)
        tree = fit_decision_tree(X_bootstrap, Y_bootstrap, attributes, max_depth)
        forest.append(tree)
    return forest

# Define a function to predict using a decision tree
def predict(tree, x):
    if isinstance(tree, dict):
        attribute = tree["attribute"]
        value = x[attribute]
        if value in tree["trees"]:
            return predict(tree["trees"][value], x)
        else:
            return tree["label"]
    else:
        return tree

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Modify the fit_decision_tree function to accept attribute list
def fit_decision_tree(X, Y, attribute_list, max_depth=None):
    decision_tree = ID3(X, Y, attribute_list, max_depth)
    return decision_tree

def adaboost_train(train_data, test_data, attributes, target, T):
    weights = np.ones(len(train_data)) / len(train_data)
    stumps = []
    train_errors = []
    test_errors = []

    for t in range(T):
        # Learn decision stump
        chosen_attr, threshold = decision_stump(train_data, weights, attributes, target)

        # Make predictions
        predictions = 2 * (train_data[target] == 1) - 1
        stump_preds = predictions != (2 * (train_data[chosen_attr] <= threshold) - 1)

        # Calculate error
        error = np.sum(weights[stump_preds])

        # Update weights
        alpha = 0.5 * np.log((1 - error) / error)
        weights *= np.exp(-alpha * stump_preds)
        weights /= np.sum(weights)

        # Evaluate train and test error
        train_preds = adaboost_predict(train_data, stumps)
        test_preds = adaboost_predict(test_data, stumps)
        train_correct = np.sum(train_preds == (2 * (train_data[target] == 1) - 1))
        test_correct = np.sum(test_preds == (2 * (test_data[target] == 1) - 1))
        train_error = 1 - train_correct / len(train_data)
        test_error = 1 - test_correct / len(test_data)

        # Store errors
        train_errors.append(train_error)
        test_errors.append(test_error)

        # Store stump
        stumps.append((chosen_attr, threshold, alpha))

    return stumps, train_errors, test_errors

# Load the credit default dataset
credit_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW2\\credit\\ccc.csv")

credit_train = credit_data.sample(n=24000, random_state=42)
credit_test = credit_data.drop(credit_train.index)

# Reset the index of the credit_train data frame
credit_train.reset_index(drop=True, inplace=True)

# In the main code section:
train_data = credit_train.copy()
test_data = credit_test.copy()


# Initialize parameters
num_trees = 500  # Number of iterations

max_depth = None  # Maximum depth of the decision trees (set to None for no maximum depth)

# Dataset column names 
attributes = list(credit_train.columns)[:-1]  # Exclude the 'y' column as the target
target = 'y'

X_train = credit_train.drop('y', axis=1).values
y_train = credit_train['y']

X_test = credit_test.drop('y', axis=1).values
y_test = credit_test['y']


# Lists to store errors
train_errors_bagging = []
test_errors_bagging = []
train_errors_forest = []
test_errors_forest = []
train_errors_adaboost = []
test_errors_adaboost = []


for t in range(T):
    # Bagged Trees
    bagged_predictions = bagged_trees(credit_train, y_train, num_trees)
    train_error_bagging = 1 - accuracy(y_train, bagged_predictions)
    test_predictions = bagged_trees(credit_test, y_train, num_trees)
    test_error_bagging = 1 - accuracy(y_test, test_predictions)
    train_errors_bagging.append(train_error_bagging)
    test_errors_bagging.append(test_error_bagging)

    # Random Forest
    forest = random_forest(credit_train, y_train, num_trees, feature_subset_size, max_depth)
    train_predictions_rf = predict_forest(forest, credit_train)
    test_predictions_rf = predict_forest(forest, credit_test)
    train_error_rf = 1 - accuracy(y_train, train_predictions_rf)
    test_error_rf = 1 - accuracy(y_test, test_predictions_rf)
    train_errors_forest.append(train_error_rf)
    test_errors_forest.append(test_error_rf)

    # Adaboost with Decision Stumps (Assuming you have Adaboost with stumps implementation)
    stumps, train_errors_ada, test_errors_ada = adaboost_train(credit_train, attributes, target, T)

    train_error_ada = train_errors_ada[t]  # Training error for the current iteration
    test_error_ada = test_errors_ada[t]  # Testing error for the current iteration
    train_errors_adaboost.append(train_error_ada)
    test_errors_adaboost.append(test_error_ada)

# Plotting the errors
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(T), train_errors_bagging, label='Train Error (Bagged Trees)')
plt.plot(range(T), test_errors_bagging, label='Test Error (Bagged Trees)')
plt.plot(range(T), train_errors_forest, label='Train Error (Random Forest)')
plt.plot(range(T), test_errors_forest, label='Test Error (Random Forest)')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations (Bagged Trees and Random Forest)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(T), train_errors_adaboost, label='Train Error (Adaboost)')
plt.plot(range(T), test_errors_adaboost, label='Test Error (Adaboost)')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error vs Iterations (Adaboost)')
plt.legend()

plt.tight_layout()
plt.show()

