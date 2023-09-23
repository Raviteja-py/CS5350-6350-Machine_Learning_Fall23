from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import mode

# Load train and test data sets
train_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW1\\bank-4\\train.csv")
test_data = pd.read_csv("C:\\Users\\ravit\\OneDrive\\Desktop\\Fall 2023\\ML\\HW1\\bank-4\\test.csv")


# print the given labels
attribute_names = train_data.columns
print("Labels:", attribute_names)

# Entropy function
def entropy(y):
    if len(y) == 0:
        return 0
    p = np.array(list(Counter(y).values())) / len(y)
    return -np.sum(p * np.log2(p))

# Information Gain Function
def information_gain(y, subsets):
    total_entropy = entropy(y)
    weighted_entropy = sum((len(subset) / len(y)) * entropy(subset) for subset in subsets)
    return total_entropy - weighted_entropy

# Majority Error Function
def majority_error(y, subsets):
    total_samples = len(y)
    majority_error_sum = sum((1 - max(Counter(subset).values()) / len(subset)) * len(subset) / total_samples for subset in subsets)
    return majority_error_sum

# Gini Index Function
def gini_index(y, subsets):
    total_samples = len(y)
    gini_index_sum = sum((1 - sum((v / len(subset)) ** 2 for v in Counter(subset).values())) * len(subset) / total_samples for subset in subsets)
    return gini_index_sum

# ID3 Implementation
def ID3(data, attributes, target_attribute, depth, IG_variants):
    # ("maxDepth cannot be lower than 1! Setting it to 1.")
    if depth == 1 or not attributes:
        target_values = data[target_attribute].values
        return mode(target_values)[0][0]

    target_values = data[target_attribute].values

    if len(set(target_values)) == 1:
        return target_values[0]

    best_attribute = None
    best_score = None

    for attribute in attributes:
        if attribute != target_attribute:
            for value in data[attribute].unique():
                subset = data[data[attribute] == value]
                if not subset.empty:
                    if IG_variants == "information_gain":
                        score = information_gain(target_values, [subset[target_attribute].values])
                    elif IG_variants == "majority_error":
                        score = majority_error(target_values, [subset[target_attribute].values])
                    elif IG_variants == "gini_index":
                        score = gini_index(target_values, [subset[target_attribute].values])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_attribute = attribute

    if best_score is None:
        return mode(target_values)[0][0]

    tree = {best_attribute: {}}
    remaining_attributes = attributes.copy()
    remaining_attributes.remove(best_attribute)

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if not subset.empty:
            subtree = ID3(subset, remaining_attributes, target_attribute, depth - 1, IG_variants)
            tree[best_attribute][value] = subtree

    return tree

# function to predict a single example
def predict_example(example, decision_tree):
    if isinstance(decision_tree, str):
        return decision_tree
    if isinstance(decision_tree, dict):
        attr = list(decision_tree.keys())[0]
        value = example[attr]
        if value in decision_tree[attr]:
            subtree = decision_tree[attr][value]
            return predict_example(example, subtree)
        else:
            return mode(train_data['y'].values)[0][0]

# function to predict a list of examples
def predict_examples(examples, decision_tree):
    return [predict_example(example, decision_tree) for example in examples]

# Indicateing Depth
max_depths = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
IG_variants = ["information_gain", "majority_error", "gini_index"]

results = []
# Training and Testing
for depth in max_depths:
    for variants in IG_variants:
        attributes_copy = list(attribute_names)
        tree = ID3(train_data, attributes_copy, 'y', depth, variants)

        train_predictions = predict_examples(train_data.to_dict(orient='records'), tree)
        test_predictions = predict_examples(test_data.to_dict(orient='records'), tree)

        train_correct = sum(1 for i in range(len(train_data)) if train_data.iloc[i]['y'] == train_predictions[i])
        test_correct = sum(1 for i in range(len(test_data)) if test_data.iloc[i]['y'] == test_predictions[i])
        train_error = 1 - (train_correct / len(train_data))
        test_error = 1 - (test_correct / len(test_data))

        results.append((depth, variants, train_error, test_error))

# Print Results
print("{:<10} {:<20} {:<15} {:<20}".format("Depth", "IG_variants", "Train Error", "Test Error"))
for result in results:
    print("{:<10} {:<20} {:<15.4f} {:<20.4f}".format(result[0], result[1], result[2], result[3]))
