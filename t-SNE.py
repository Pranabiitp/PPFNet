#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Load models with compile=False
model1 = load_model("non-iid_densenet_adaptive_dp.h5", compile=False)
model2 = load_model("non-iid_resnet_adaptive_dp.h5", compile=False)
model3 = load_model("non-iid_MobileNetV2_adaptive_dp.h5", compile=False)

# Compile each model
optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate as needed
loss_function = CategoricalCrossentropy()

model1.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model2.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model3.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Load and preprocess the test data
test = np.load("test.npy")
test = test / 255

# Load the one-hot encoded labels
label = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(label, axis=1)
unique_classes = np.unique(categorical_labels)
print(f"Unique classes in the labels: {unique_classes}")

# Function to get embeddings
def get_embeddings(model, data, layer_name='dense_303'):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)

# Get embeddings
embeddings = get_embeddings(model1, test)

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot t-SNE visualization
plt.figure(figsize=(10, 8))

# Iterate over unique classes and plot
for class_label in unique_classes:
    idx = np.where(categorical_labels == class_label)
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=str(class_label))

plt.title('t-SNE Visualization of Model Embeddings')
plt.legend(title='Classes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
plt.savefig('tsne1.png', bbox_inches='tight', dpi=300)
plt.show()


# In[ ]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model

# Load the model with compile=False
model2 = load_model("non-iid_resnet_adaptive_dp.h5", compile=False)

# Load and preprocess the test data
test = np.load("test.npy")
test = test / 255

# Load the one-hot encoded labels
label = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(label, axis=1)
unique_classes = np.unique(categorical_labels)
print(f"Unique classes in the labels: {unique_classes}")

# Function to get embeddings
def get_embeddings(model, data, layer_name='dense_605'):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)

# Get embeddings for the test data
embeddings = get_embeddings(model2, test)

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot t-SNE visualization
plt.figure(figsize=(10, 8))

# Iterate over unique classes and plot
for class_label in unique_classes:
    idx = np.where(categorical_labels == class_label)
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=str(class_label))

plt.title('t-SNE Visualization of Model Embeddings')
plt.legend(title='Classes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
plt.savefig('tsne2.png', bbox_inches='tight', dpi=300)
plt.show()


# In[ ]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model

# Load the model with compile=False
model3 = load_model("non-iid_MobileNetV2_adaptive_dp.h5", compile=False)

# Load and preprocess the test data
test = np.load("test.npy")
test = test / 255

# Load the one-hot encoded labels
label = np.load("one_hot_labels.npy")

# Convert one-hot encoded labels to categorical labels
categorical_labels = np.argmax(label, axis=1)
unique_classes = np.unique(categorical_labels)
print(f"Unique classes in the labels: {unique_classes}")

# Function to get embeddings
def get_embeddings(model, data, layer_name='dense_1'):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)

# Get embeddings for the test data
embeddings = get_embeddings(model3, test)

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot t-SNE visualization
plt.figure(figsize=(10, 8))

# Iterate over unique classes and plot
for class_label in unique_classes:
    idx = np.where(categorical_labels == class_label)
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=str(class_label))

plt.title('t-SNE Visualization of Model Embeddings')
plt.legend(title='Classes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
plt.savefig('tsne3.png', bbox_inches='tight', dpi=300)
plt.show()


# In[ ]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import LabelEncoder

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
unique_classes = np.unique(categorical_labels)
print(f"Unique classes in the labels: {unique_classes}")

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
accuracies = [0.814, 0.804, 0.798]  # Example accuracies for Model 1, Model 2, Model 3

# Calculate weights using power weighting function
weights = [accuracy * (s_correct ** gamma)
           for accuracy, s_correct in zip(accuracies, s_correct_values)]

# Normalize weights
weights = np.array(weights) / np.sum(weights)

# Ensemble predictions
ensemble_preds = weights[0] * preds1 + weights[1] * preds2 + weights[2] * preds3

# Convert probabilities to class labels
ensemble_labels = np.argmax(ensemble_preds, axis=1)

# Function to get embeddings from each model
def get_embeddings(model, data, layer_name):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(data)

# Get embeddings for each model
embeddings1 = get_embeddings(model1, test_data, layer_name='dense_303')
embeddings2 = get_embeddings(model2, test_data, layer_name='dense_605')
embeddings3 = get_embeddings(model3, test_data, layer_name='dense_1')

# Aggregate embeddings (e.g., by averaging)
aggregated_embeddings = (embeddings1 + embeddings2 + embeddings3) / 3

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(aggregated_embeddings)

# Plot t-SNE visualization
plt.figure(figsize=(10, 8))

# Iterate over unique classes and plot
for class_label in unique_classes:
    idx = np.where(ensemble_labels == class_label)
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=str(class_label))

plt.title('t-SNE Visualization of Ensemble Model Embeddings')
plt.legend(title='Classes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
plt.savefig('tsne_ensemble.png', bbox_inches='tight', dpi=300)
plt.show()

