#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# In[1]:





# In[5]:





# In[ ]:





# In[22]:


from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Load models with compile=False
model1 = load_model("non-iid_densenet_adaptive_dp.h5", compile=False)
model2 = load_model("non-iid_resnet_adaptive_dp.h5", compile=False)
model3 = load_model("non-iid_MobileNetV2_adaptive_dp.h5", compile=False)

# Compile each model
optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate as needed
loss_function =CategoricalCrossentropy()

model1.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model2.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model3.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

import numpy as np
test=np.load("test.npy")
label=np.load("one_hot_labels.npy")
test=test/255


# In[ ]:


test_data=test
true_labels=label
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data and labels
# test_data = np.load('test.npy')
# true_labels = np.load('label.npy')

# # Load models
# model1 = tf.keras.models.load_model('model1.h5')
# model2 = tf.keras.models.load_model('model2.h5')
# model3 = tf.keras.models.load_model('model3.h5')

# Make predictions for each model
preds1 = model1.predict(test_data)
preds2 = model2.predict(test_data)
preds3 = model3.predict(test_data)
acc=[]
# Extract predicted probabilities for correct class
s_correct_values = []
for preds, true_label in zip([preds1, preds2, preds3], true_labels):
    correct_class_prob = preds[np.arange(len(true_label)), true_label]
    s_correct_values.append(np.mean(correct_class_prob))

gamma_values = np.linspace(1.0, 20.0, 10)

# Define accuracies for each model
accuracies = [.8260, .8140,.8040]  # Example accuracies for Model 1, Model 2, Model 3
for gamma in gamma_values:
# Calculate weights using power weighting function
 weights = [accuracy * (s_correct ** gamma)
           for accuracy, s_correct in zip(accuracies, s_correct_values)]

# Normalize weights
 weights = np.array(weights) / np.sum(weights)

# Ensemble predictions
 ensemble_preds = weights[0] * preds1 + weights[1] * preds2 + weights[2] * preds3

# Convert probabilities to class labels
 ensemble_labels = np.argmax(ensemble_preds, axis=1)
 from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Convert ensemble labels to integer format
 label_encoder = LabelEncoder()
 ensemble_labels_int = label_encoder.fit_transform(ensemble_labels)

# Convert integer labels to one-hot encoding
 onehot_encoder = OneHotEncoder(sparse=False)
 ensemble_labels = onehot_encoder.fit_transform(ensemble_labels_int.reshape(-1, 1))
# Evaluate ensemble performance
 accuracy = accuracy_score(true_labels, ensemble_labels)
 acc.append(accuracy)
#  precision = precision_score(true_labels, ensemble_labels, average='weighted')
#  recall = recall_score(true_labels, ensemble_labels, average='weighted')
#  f1 = f1_score(true_labels, ensemble_labels, average='weighted')

#  print("Ensemble Performance:")
#  print("Accuracy:", accuracy)
#  print("Precision:", precision)
#  print("Recall:", recall)
#  print("F1 Score:", f1)
plt.plot(acc)
plt.grid(visible=True)


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


# In[ ]:





# In[ ]:





# In[25]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:





# In[5]:





# In[ ]:





# In[6]:





# In[86]:





# In[ ]:





# In[ ]:





# In[64]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


accuracy_weighted=[]
precision_weighted=[]
recall_weighted=[]
fscore_weighted=[]
accuracy_maj_voting=[]
precision_maj_voting=[]
recall_maj_voting=[]
fscore_maj_voting=[]
def majority_voting(true_label,model1_labels, model1_accuracy, model2_labels, model2_accuracy, model3_labels, model3_accuracy):
    model1_labels = np.argmax(model1_labels, axis=1)
    model2_labels = np.argmax(model2_labels, axis=1)
    model3_labels = np.argmax(model3_labels, axis=1)
    combined_preds = np.array([model1_labels, model2_labels, model3_labels])

    # Take the mode of each prediction across the three models
    final_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=combined_preds)
    predictions = final_preds
    y_label=true_label
    y_label=np.argmax(y_label, axis=1)
    print("Accuracy of majority voting:",accuracy_score(predictions,y_label))
    accuracy_maj_voting.append(accuracy_score(predictions,y_label))
    print("macro precision score of majority voting:",metrics.precision_score(predictions,y_label,average='macro'))
    precision_maj_voting.append(metrics.precision_score(predictions,y_label,average='macro'))
    print("macro recall score of majority voting:",metrics.recall_score(predictions,y_label,average='macro'))
    recall_maj_voting.append(metrics.recall_score(predictions,y_label,average='macro'))
    print("macro f1 score of majority voting:",f1_score(predictions, y_label, average='macro'))
    fscore_maj_voting.append(f1_score(predictions, y_label, average='macro'))

    
accuracy_averaging=[]
precision_averaging=[]
recall_averaging=[]
fscore_averaging=[]
def averaging(true_label,model1_labels, model1_accuracy, model2_labels, model2_accuracy, model3_labels, model3_accuracy):
    combined_preds = (model1_labels+ model2_labels+ model3_labels)/3
    final_preds = np.argmax(combined_preds, axis=1)
    predictions = final_preds
    y_label=true_label
    y_label=np.argmax(y_label, axis=1)
    print("Accuracy of averaging:",accuracy_score(predictions,y_label))
    accuracy_averaging.append(accuracy_score(predictions,y_label))
    print("macro precision score of averaging:",metrics.precision_score(predictions,y_label,average='macro'))
    precision_averaging.append(metrics.precision_score(predictions,y_label,average='macro'))
    print("macro recall score of averaging:",metrics.recall_score(predictions,y_label,average='macro'))
    recall_averaging.append(metrics.recall_score(predictions,y_label,average='macro'))
    print("macro f1 score of averaging:",f1_score(predictions, y_label, average='macro'))
    fscore_averaging.append(f1_score(predictions, y_label, average='macro'))
    
def weighted_average(true_label,model1_labels, model1_accuracy, model2_labels, model2_accuracy, model3_labels, model3_accuracy):
    n_classes=5
    model1_binary = model1_labels
    model2_binary = model2_labels
    model3_binary = model3_labels
    model1_weight = model1_accuracy/(model1_accuracy+model2_accuracy+model3_accuracy)
    model2_weight = model2_accuracy/(model1_accuracy+model2_accuracy+model3_accuracy)
    model3_weight = model3_accuracy/(model1_accuracy+model2_accuracy+model3_accuracy)
#     print(model1_binary.shape)
#     print(model2_binary,model3_binary)
    
    weighted_average = (model1_weight * model1_binary + 
                    model2_weight * model2_binary +
                    model3_weight * model3_binary)
    predicted_labels = np.argmax(weighted_average, axis=1)
    predictions=predicted_labels
    y_label=true_label
    y_label=np.argmax(y_label, axis=1)
    print("Accuracy of weighted average is:",accuracy_score(predictions,y_label))
    accuracy_weighted.append(accuracy_score(predictions,y_label))
    print("micro precision score of weighted average:",metrics.precision_score(predictions,y_label,average='micro'))
    print("macro precision score of weighted average:",metrics.precision_score(predictions,y_label,average='macro'))
    precision_weighted.append(metrics.precision_score(predictions,y_label,average='macro'))
    print("micro recall score of weighted average:",metrics.recall_score(predictions,y_label,average='micro'))
    print("macro recall score of weighted average:",metrics.recall_score(predictions,y_label,average='macro'))
    recall_weighted.append(metrics.recall_score(predictions,y_label,average='macro'))
    print("micro f1 score of weighted average:",f1_score(predictions, y_label, average='micro'))
    print("macro f1 score of weighted average:",f1_score(predictions, y_label, average='macro'))
    fscore_weighted.append(f1_score(predictions, y_label, average='macro'))


# In[ ]:





# In[89]:


label = np.argmax(label, axis=1)


# In[14]:


correct=0
pred_dense=model1.predict(test)
pred_dense = np.argmax(pred_dense, axis=1)
# print(pred_dense)
for i in range(len(label)):
                if label[i]==pred_dense[i]: 
                    
                    
                    correct+=1
acc_dense=(correct/len(label))


# In[13]:


acc_dense


# In[11]:


correct=0
pred_resnet=model2.predict(test)
pred_resnet = np.argmax(pred_resnet, axis=1)
for i in range(len(label)):
                
                if label[i]==pred_resnet[i]: 
                    
                   
                    correct+=1
acc_resnet=(correct/len(label))
acc_resnet


# In[12]:


correct=0
pred_inception=model3.predict(test)
pred_inception = np.argmax(pred_inception, axis=1)
for i in range(len(label)):
                
                if label[i]==pred_inception[i]: 
                    
                   
                    correct+=1
acc_inception=(correct/len(label))
acc_inception


# In[94]:


import numpy as np

# Your integer labels


# Find the number of unique labels
num_classes = len(np.unique(label))

# Initialize an empty one-hot encoding matrix with zeros
one_hot_matrix = np.zeros((len(label), num_classes))

# Fill the matrix with 1s in the appropriate columns
for i, label in enumerate(label):
    one_hot_matrix[i, label] = 1

# Print the one-hot encoded matrix
# print(one_hot_matrix)


# In[10]:


# one_hot_matrix.shape


# In[96]:


test=test/255


# In[97]:





# In[6]:


pred_dense=model1.predict(test)
pred_resnet=model2.predict(test)


pred_inception=model3.predict(test)


# In[3]:





# In[100]:





# In[101]:





# In[4]:





# In[5]:


weighted_average(one_hot_matrix,pred_dense, acc_dense, pred_resnet, acc_resnet, pred_inception,acc_inception)
majority_voting(one_hot_matrix,pred_dense, acc_dense, pred_resnet, acc_resnet, pred_inception,acc_inception)
averaging(one_hot_matrix,pred_dense, acc_dense, pred_resnet, acc_resnet, pred_inception,acc_inception)

            


# In[81]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


#sigmoid


# In[ ]:





# In[ ]:





# In[46]:


from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Load models with compile=False
model1 = load_model("unbalanced_densenet_adaptive_DP.h5", compile=False)
model2 = load_model("unbalanced_ResNet101V2_adaptive_DP.h5", compile=False)
model3 = load_model("unbalanced_mobilenet_adaptive_DP.h5", compile=False)

# Compile each model
optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate as needed
loss_function =CategoricalCrossentropy()

model1.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model2.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model3.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

import numpy as np
test=np.load("test.npy")
label=np.load("one_hot_labels.npy")
test=test/255


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:





# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




