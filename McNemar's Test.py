#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from tensorflow.keras.models import load_model

# Load the models with compile=False
model1 = load_model("unbalanced_densenet_adaptive_DP.h5", compile=False)
model2 = load_model("unbalanced_ResNet101V2_adaptive_DP.h5", compile=False)
model3 = load_model("unbalanced_mobilenet_adaptive_DP.h5", compile=False)

# Load and preprocess the test data
test_data = np.load("test.npy")
test_data = test_data / 255

# Load the one-hot encoded labels
true_labels = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(true_labels, axis=1)

# Make predictions for each model
preds1 = model1.predict(test_data)
preds2 = model2.predict(test_data)
preds3 = model3.predict(test_data)

# Extract predicted probabilities for correct class
s_correct_values = []
for preds in [preds1, preds2, preds3]:
    correct_class_prob = preds[np.arange(len(categorical_labels)), categorical_labels]
    s_correct_values.append(np.mean(correct_class_prob))

# Define gamma for power weighting function
gamma = 1.1  # Example value, can be adjusted

# Define accuracies for each model
accuracies = [.8260, .8140 , .8040]  # Example accuracies for Model 1, Model 2, Model 3

# Calculate weights using power weighting function
weights = [accuracy * (s_correct ** gamma)
           for accuracy, s_correct in zip(accuracies, s_correct_values)]

# Normalize weights
weights = np.array(weights) / np.sum(weights)

# Ensemble predictions
ensemble_preds = weights[0] * preds1 + weights[1] * preds2 + weights[2] * preds3

# Convert probabilities to class labels
ensemble_labels = np.argmax(ensemble_preds, axis=1)

# Convert predictions to binary values (1 for correct, 0 for incorrect)
correct_model1 = (preds1.argmax(axis=1) == categorical_labels).astype(int)
correct_model2 = (preds2.argmax(axis=1) == categorical_labels).astype(int)
correct_model3 = (preds3.argmax(axis=1) == categorical_labels).astype(int)
correct_ensemble = (ensemble_labels == categorical_labels).astype(int)

# Function to perform chi-squared test
def chi_squared_test(correct_model, correct_ensemble):
    # Create a contingency table
    contingency_table = np.zeros((2, 2), dtype=int)
    contingency_table[0, 0] = np.sum((correct_model == 1) & (correct_ensemble == 1))
    contingency_table[0, 1] = np.sum((correct_model == 0) & (correct_ensemble == 1))
    contingency_table[1, 0] = np.sum((correct_model == 1) & (correct_ensemble == 0))
    contingency_table[1, 1] = np.sum((correct_model == 0) & (correct_ensemble == 0))

    # Perform chi-squared test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    return chi2, p_value

# Perform chi-squared tests
chi2_model1, p_value_model1 = chi_squared_test(correct_model1, correct_ensemble)
chi2_model2, p_value_model2 = chi_squared_test(correct_model2, correct_ensemble)
chi2_model3, p_value_model3 = chi_squared_test(correct_model3, correct_ensemble)

# Print the results
print(f"Model 1 vs Ensemble - Chi-squared Statistic: {chi2_model1}, P-value: {p_value_model1}")
print(f"Model 2 vs Ensemble - Chi-squared Statistic: {chi2_model2}, P-value: {p_value_model2}")
print(f"Model 3 vs Ensemble - Chi-squared Statistic: {chi2_model3}, P-value: {p_value_model3}")

# Compare with significance level (e.g., 0.05)
alpha = 0.05
for i, p_value in enumerate([p_value_model1, p_value_model2, p_value_model3], 1):
    if p_value < alpha:
        print(f"Reject the null hypothesis for Model {i}: There is a significant difference in performance.")
    else:
        print(f"Fail to reject the null hypothesis for Model {i}: No significant difference in performance.")

