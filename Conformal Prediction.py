#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

# Set font size and weight
rcParams['font.size'] = 14
rcParams['font.weight'] = 'bold'

# Load the models with compile=False
model1 = load_model("non-iid_densenet_adaptive_dp.h5", compile=False)
model2 = load_model("non-iid_resnet_adaptive_dp.h5", compile=False)
model3 = load_model("non-iid_MobileNetV2_adaptive_dp.h5", compile=False)

# Load and preprocess the test data
test_data = np.load("test.npy")
test_data = test_data / 255

# Load the one-hot encoded labels
true_labels = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(true_labels, axis=1)

# Split the test data into calibration and validation sets
calibration_data, validation_data, calibration_labels, validation_labels = train_test_split(
    test_data, categorical_labels, test_size=0.5, random_state=42)

# Define gamma for power weighting function
gamma = 1.1  # Example value, can be adjusted

# Define accuracies for each model
accuracies = [0.814, 0.804, 0.798]  # Example accuracies for Model 1, Model 2, Model 3

# Function to calculate nonconformity scores
def nonconformity(preds, true_labels):
    return 1 - preds[np.arange(len(true_labels)), true_labels]

# Function to generate prediction sets
def prediction_set(preds, quantile):
    prediction_sets = []
    for i in range(len(preds)):
        pred_set = np.where(preds[i] >= 1 - quantile)[0]
        prediction_sets.append(pred_set)
    return prediction_sets

# Function to evaluate prediction sets
def evaluate_prediction_sets(prediction_sets, true_labels):
    correct_predictions = 0
    for i in range(len(true_labels)):
        if true_labels[i] in prediction_sets[i]:
            correct_predictions += 1
    coverage = correct_predictions / len(true_labels)
    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
    return coverage, avg_set_size

# Models' predictions on calibration data
cal_preds1 = model1.predict(calibration_data)
cal_preds2 = model2.predict(calibration_data)
cal_preds3 = model3.predict(calibration_data)

# Extract predicted probabilities for correct class on calibration data
cal_s_correct_values = []
for cal_preds in [cal_preds1, cal_preds2, cal_preds3]:
    correct_class_prob = cal_preds[np.arange(len(calibration_labels)), calibration_labels]
    cal_s_correct_values.append(correct_class_prob)

# Calculate weights using power weighting function
weights = [accuracy * (np.mean(s_correct) ** gamma)
           for accuracy, s_correct in zip(accuracies, cal_s_correct_values)]

# Normalize weights
weights = np.array(weights) / np.sum(weights)

# Ensemble predictions on calibration data
cal_ensemble_preds = weights[0] * cal_preds1 + weights[1] * cal_preds2 + weights[2] * cal_preds3

# Calculate nonconformity scores for each model
nonconformity_scores1 = nonconformity(cal_preds1, calibration_labels)
nonconformity_scores2 = nonconformity(cal_preds2, calibration_labels)
nonconformity_scores3 = nonconformity(cal_preds3, calibration_labels)
nonconformity_scores_ensemble = nonconformity(cal_ensemble_preds, calibration_labels)

# Prepare data for plotting
confidence_levels = np.arange(0.05, 1.0, 0.05)
coverages = {"model1": [], "model2": [], "model3": [], "ensemble": []}
set_sizes = {"model1": [], "model2": [], "model3": [], "ensemble": []}

for conf_level in confidence_levels:
    quantile1 = np.quantile(nonconformity_scores1, conf_level)
    quantile2 = np.quantile(nonconformity_scores2, conf_level)
    quantile3 = np.quantile(nonconformity_scores3, conf_level)
    quantile_ensemble = np.quantile(nonconformity_scores_ensemble, conf_level)

    # Calculate prediction sets for validation data
    val_preds1 = model1.predict(validation_data)
    val_preds2 = model2.predict(validation_data)
    val_preds3 = model3.predict(validation_data)
    val_ensemble_preds = weights[0] * val_preds1 + weights[1] * val_preds2 + weights[2] * val_preds3

    pred_sets1 = prediction_set(val_preds1, quantile1)
    pred_sets2 = prediction_set(val_preds2, quantile2)
    pred_sets3 = prediction_set(val_preds3, quantile3)
    pred_sets_ensemble = prediction_set(val_ensemble_preds, quantile_ensemble)

    # Evaluate coverage and average set size for each model
    coverage1, avg_size1 = evaluate_prediction_sets(pred_sets1, validation_labels)
    coverage2, avg_size2 = evaluate_prediction_sets(pred_sets2, validation_labels)
    coverage3, avg_size3 = evaluate_prediction_sets(pred_sets3, validation_labels)
    coverage_ensemble, avg_size_ensemble = evaluate_prediction_sets(pred_sets_ensemble, validation_labels)

    coverages["model1"].append(coverage1)
    coverages["model2"].append(coverage2)
    coverages["model3"].append(coverage3)
    coverages["ensemble"].append(coverage_ensemble)

    set_sizes["model1"].append(avg_size1)
    set_sizes["model2"].append(avg_size2)
    set_sizes["model3"].append(avg_size3)
    set_sizes["ensemble"].append(avg_size_ensemble)

# Plot Average Prediction Set Size vs. Coverage
plt.figure(figsize=(10, 6))
plt.plot(set_sizes["model1"], coverages["model1"], label="Model 1")
plt.plot(set_sizes["model2"], coverages["model2"], label="Model 2")
plt.plot(set_sizes["model3"], coverages["model3"], label="Model 3")
plt.plot(set_sizes["ensemble"], coverages["ensemble"], label="Ensemble", linestyle='--', linewidth=2)
plt.ylabel("Coverage", fontweight='bold')
plt.xlabel("Average Prediction Set Size", fontweight='bold')
# plt.title("Average Prediction Set Size vs. Coverage", fontweight='bold')
plt.legend()
plt.savefig("non-iid_conf", dpi=350)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

# Set font size and weight
rcParams['font.size'] = 14
rcParams['font.weight'] = 'bold'

# Load the models with compile=False
model1 = load_model("iid_densenet_adaptive_DP.h5", compile=False)
model2 = load_model("iid_resnet_adaptive_DP.h5", compile=False)
model3 = load_model("iid_moblenetV2_adaptive_DP.h5", compile=False)

# Load and preprocess the test data
test_data = np.load("test.npy")
test_data = test_data / 255

# Load the one-hot encoded labels
true_labels = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(true_labels, axis=1)

# Split the test data into calibration and validation sets
calibration_data, validation_data, calibration_labels, validation_labels = train_test_split(
    test_data, categorical_labels, test_size=0.5, random_state=42)

# Define gamma for power weighting function
gamma = 1.1

# Define accuracies for each model
accuracies = [.8890, .8450, .8510]

# Function to calculate nonconformity scores
def nonconformity(preds, true_labels):
    return 1 - preds[np.arange(len(true_labels)), true_labels]

# Function to generate prediction sets
def prediction_set(preds, quantile):
    prediction_sets = []
    for i in range(len(preds)):
        pred_set = np.where(preds[i] >= 1 - quantile)[0]
        prediction_sets.append(pred_set)
    return prediction_sets

# Function to evaluate prediction sets
def evaluate_prediction_sets(prediction_sets, true_labels):
    correct_predictions = 0
    for i in range(len(true_labels)):
        if true_labels[i] in prediction_sets[i]:
            correct_predictions += 1
    coverage = correct_predictions / len(true_labels)
    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
    return coverage, avg_set_size

# Models' predictions on calibration data
cal_preds1 = model1.predict(calibration_data)
cal_preds2 = model2.predict(calibration_data)
cal_preds3 = model3.predict(calibration_data)

# Extract predicted probabilities for correct class on calibration data
cal_s_correct_values = []
for cal_preds in [cal_preds1, cal_preds2, cal_preds3]:
    correct_class_prob = cal_preds[np.arange(len(calibration_labels)), calibration_labels]
    cal_s_correct_values.append(correct_class_prob)

# Calculate weights using power weighting function
weights = [accuracy * (np.mean(s_correct) ** gamma)
           for accuracy, s_correct in zip(accuracies, cal_s_correct_values)]

# Normalize weights
weights = np.array(weights) / np.sum(weights)

# Ensemble predictions on calibration data
cal_ensemble_preds = weights[0] * cal_preds1 + weights[1] * cal_preds2 + weights[2] * cal_preds3

# Calculate nonconformity scores for each model
nonconformity_scores1 = nonconformity(cal_preds1, calibration_labels)
nonconformity_scores2 = nonconformity(cal_preds2, calibration_labels)
nonconformity_scores3 = nonconformity(cal_preds3, calibration_labels)
nonconformity_scores_ensemble = nonconformity(cal_ensemble_preds, calibration_labels)

# Prepare data for plotting
confidence_levels = np.arange(0.05, 1.0, 0.05)
coverages = {"model1": [], "model2": [], "model3": [], "ensemble": []}
set_sizes = {"model1": [], "model2": [], "model3": [], "ensemble": []}

for conf_level in confidence_levels:
    quantile1 = np.quantile(nonconformity_scores1, conf_level)
    quantile2 = np.quantile(nonconformity_scores2, conf_level)
    quantile3 = np.quantile(nonconformity_scores3, conf_level)
    quantile_ensemble = np.quantile(nonconformity_scores_ensemble, conf_level)

    # Calculate prediction sets for validation data
    val_preds1 = model1.predict(validation_data)
    val_preds2 = model2.predict(validation_data)
    val_preds3 = model3.predict(validation_data)
    val_ensemble_preds = weights[0] * val_preds1 + weights[1] * val_preds2 + weights[2] * val_preds3

    pred_sets1 = prediction_set(val_preds1, quantile1)
    pred_sets2 = prediction_set(val_preds2, quantile2)
    pred_sets3 = prediction_set(val_preds3, quantile3)
    pred_sets_ensemble = prediction_set(val_ensemble_preds, quantile_ensemble)

    # Evaluate coverage and average set size for each model
    coverage1, avg_size1 = evaluate_prediction_sets(pred_sets1, validation_labels)
    coverage2, avg_size2 = evaluate_prediction_sets(pred_sets2, validation_labels)
    coverage3, avg_size3 = evaluate_prediction_sets(pred_sets3, validation_labels)
    coverage_ensemble, avg_size_ensemble = evaluate_prediction_sets(pred_sets_ensemble, validation_labels)

    coverages["model1"].append(coverage1)
    coverages["model2"].append(coverage2)
    coverages["model3"].append(coverage3)
    coverages["ensemble"].append(coverage_ensemble)

    set_sizes["model1"].append(avg_size1)
    set_sizes["model2"].append(avg_size2)
    set_sizes["model3"].append(avg_size3)
    set_sizes["ensemble"].append(avg_size_ensemble)

# Plot Average Prediction Set Size vs. Coverage
plt.figure(figsize=(10, 6))
plt.plot(set_sizes["model1"], coverages["model1"], label="Model 1")
plt.plot(set_sizes["model2"], coverages["model2"], label="Model 2")
plt.plot(set_sizes["model3"], coverages["model3"], label="Model 3")
plt.plot(set_sizes["ensemble"], coverages["ensemble"], label="Ensemble", linestyle='--', linewidth=2)
plt.ylabel("Coverage", fontweight='bold')
plt.xlabel("Average Prediction Set Size", fontweight='bold')
# plt.title("Average Prediction Set Size vs. Coverage", fontweight='bold')
plt.legend()
plt.savefig("iid_conf", dpi=350)
plt.show()


# In[ ]:


# Load the models with compile=False
# model1 = load_model("non-iid_densenet_adaptive_dp.h5", compile=False)
# model2 = load_model("non-iid_resnet_adaptive_dp.h5", compile=False)
# model3 = load_model("non-iid_MobileNetV2_adaptive_dp.h5", compile=False)
model1 = load_model("unbalanced_densenet_adaptive_DP.h5", compile=False)
model2 = load_model("unbalanced_ResNet101V2_adaptive_DP.h5", compile=False)
model3 = load_model("unbalanced_mobilenet_adaptive_DP.h5", compile=False)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

# Set font size and weight
rcParams['font.size'] = 14
rcParams['font.weight'] = 'bold'



# Load and preprocess the test data
test_data = np.load("test.npy")
test_data = test_data / 255

# Load the one-hot encoded labels
true_labels = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(true_labels, axis=1)

# Split the test data into calibration and validation sets
calibration_data, validation_data, calibration_labels, validation_labels = train_test_split(
    test_data, categorical_labels, test_size=0.5, random_state=42)

# Define gamma for power weighting function
gamma = 1.1

# Define accuracies for each model
accuracies = [.8260 , .8140 , .8040 ]

# Function to calculate nonconformity scores
def nonconformity(preds, true_labels):
    return 1 - preds[np.arange(len(true_labels)), true_labels]

# Function to generate prediction sets
def prediction_set(preds, quantile):
    prediction_sets = []
    for i in range(len(preds)):
        pred_set = np.where(preds[i] >= 1 - quantile)[0]
        prediction_sets.append(pred_set)
    return prediction_sets

# Function to evaluate prediction sets
def evaluate_prediction_sets(prediction_sets, true_labels):
    correct_predictions = 0
    for i in range(len(true_labels)):
        if true_labels[i] in prediction_sets[i]:
            correct_predictions += 1
    coverage = correct_predictions / len(true_labels)
    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
    return coverage, avg_set_size

# Models' predictions on calibration data
cal_preds1 = model1.predict(calibration_data)
cal_preds2 = model2.predict(calibration_data)
cal_preds3 = model3.predict(calibration_data)

# Extract predicted probabilities for correct class on calibration data
cal_s_correct_values = []
for cal_preds in [cal_preds1, cal_preds2, cal_preds3]:
    correct_class_prob = cal_preds[np.arange(len(calibration_labels)), calibration_labels]
    cal_s_correct_values.append(correct_class_prob)

# Calculate weights using power weighting function
weights = [accuracy * (np.mean(s_correct) ** gamma)
           for accuracy, s_correct in zip(accuracies, cal_s_correct_values)]

# Normalize weights
weights = np.array(weights) / np.sum(weights)

# Ensemble predictions on calibration data
cal_ensemble_preds = weights[0] * cal_preds1 + weights[1] * cal_preds2 + weights[2] * cal_preds3

# Calculate nonconformity scores for each model
nonconformity_scores1 = nonconformity(cal_preds1, calibration_labels)
nonconformity_scores2 = nonconformity(cal_preds2, calibration_labels)
nonconformity_scores3 = nonconformity(cal_preds3, calibration_labels)
nonconformity_scores_ensemble = nonconformity(cal_ensemble_preds, calibration_labels)

# Prepare data for plotting
confidence_levels = np.arange(0.05, 1.0, 0.05)
coverages = {"model1": [], "model2": [], "model3": [], "ensemble": []}
set_sizes = {"model1": [], "model2": [], "model3": [], "ensemble": []}

for conf_level in confidence_levels:
    quantile1 = np.quantile(nonconformity_scores1, conf_level)
    quantile2 = np.quantile(nonconformity_scores2, conf_level)
    quantile3 = np.quantile(nonconformity_scores3, conf_level)
    quantile_ensemble = np.quantile(nonconformity_scores_ensemble, conf_level)

    # Calculate prediction sets for validation data
    val_preds1 = model1.predict(validation_data)
    val_preds2 = model2.predict(validation_data)
    val_preds3 = model3.predict(validation_data)
    val_ensemble_preds = weights[0] * val_preds1 + weights[1] * val_preds2 + weights[2] * val_preds3

    pred_sets1 = prediction_set(val_preds1, quantile1)
    pred_sets2 = prediction_set(val_preds2, quantile2)
    pred_sets3 = prediction_set(val_preds3, quantile3)
    pred_sets_ensemble = prediction_set(val_ensemble_preds, quantile_ensemble)

    # Evaluate coverage and average set size for each model
    coverage1, avg_size1 = evaluate_prediction_sets(pred_sets1, validation_labels)
    coverage2, avg_size2 = evaluate_prediction_sets(pred_sets2, validation_labels)
    coverage3, avg_size3 = evaluate_prediction_sets(pred_sets3, validation_labels)
    coverage_ensemble, avg_size_ensemble = evaluate_prediction_sets(pred_sets_ensemble, validation_labels)

    coverages["model1"].append(coverage1)
    coverages["model2"].append(coverage2)
    coverages["model3"].append(coverage3)
    coverages["ensemble"].append(coverage_ensemble)

    set_sizes["model1"].append(avg_size1)
    set_sizes["model2"].append(avg_size2)
    set_sizes["model3"].append(avg_size3)
    set_sizes["ensemble"].append(avg_size_ensemble)

# Plot Average Prediction Set Size vs. Coverage
plt.figure(figsize=(10, 6))
plt.plot(set_sizes["model1"], coverages["model1"], label="Model 1")
plt.plot(set_sizes["model2"], coverages["model2"], label="Model 2")
plt.plot(set_sizes["model3"], coverages["model3"], label="Model 3")
plt.plot(set_sizes["ensemble"], coverages["ensemble"], label="Ensemble", linestyle='--', linewidth=2)
plt.ylabel("Coverage", fontweight='bold')
plt.xlabel("Average Prediction Set Size", fontweight='bold')
# plt.title("Average Prediction Set Size vs. Coverage", fontweight='bold')
plt.legend()
plt.savefig("unb_conf", dpi=350)
plt.show()

