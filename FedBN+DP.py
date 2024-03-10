#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# In[2]:


import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


# In[3]:


def test_model(X_test, Y_test,  model, comm_round):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
#     logits = model.predict(X_test)
    #print(logits)
    loss,accuracy=model.evaluate(X_test,Y_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, accuracy, loss))
    return accuracy, loss


# In[4]:


import tensorflow as tf

class SimpleMLP:
    @staticmethod
    def build():
        # Build the base model (MobileNetV2)
        base_model = tf.keras.applications.ResNet101V2(
            include_top=False, weights='imagenet', input_shape=(128, 128, 3)
        )

        # Freeze the base model's layers
        base_model.trainable = False

        # Create the transfer learning model by adding custom classification layers on top of the base model
        model2 = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # Adjust the number of output classes accordingly
        ])

        # Compile the model
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model2


# In[6]:


def create_clients(data_dict):
    '''
    Return a dictionary with keys as client names and values as data and label lists.
    
    Args:
        data_dict: A dictionary where keys are client names, and values are tuples of data and labels.
                    For example, {'client_1': (data_1, labels_1), 'client_2': (data_2, labels_2), ...}
    
    Returns:
        A dictionary with keys as client names and values as tuples of data and label lists.
    '''
    return data_dict


# In[7]:


get_ipython().run_line_magic('run', 'data_augmentation_(1).ipynb')


# 

# In[8]:


test=np.load("test.npy")
one_hot_labels=np.load("one_hot_labels.npy")


# In[9]:


test=test/255


# In[11]:


np.min(test)


# In[10]:


client_data1 = {
    'client1': (test, one_hot_labels),
    'client2': (test, one_hot_labels),
    'client3': (test, one_hot_labels)
    
}
#create clients
# test_batched = create_clients(client_data1)
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3)
    
}


# In[11]:


test_batched = create_clients(client_data1)
clients_batched = create_clients(client_data2)


# In[18]:


# import tensorflow as tf
# import numpy as np

# def dp(client_model, noise_level):
#     client_params = client_model.get_weights()

#     for layer_idx in range(len(client_params)):
#         if layer_idx >= len(client_model.layers):
#             continue

#         if 'labels' in client_model.layers[layer_idx].name:
#             continue
#         else:
#             if noise_level > 0 and 'bias' not in client_model.layers[layer_idx].name:
#                 noise = noise_level * np.random.normal(
#                     loc=0,
#                     scale=np.std(client_params[layer_idx].ravel())
#                 )
#                 client_params[layer_idx] += noise

#     client_model.set_weights(client_params)
#     return client_model


# In[13]:


# import tensorflow as tf
# import numpy as np

# def dp(client_model, noise_level):
#     client_params = client_model.get_weights()

#     for layer_idx in range(min(len(client_params), len(client_model.layers))):
#         try:
#             layer_name = client_model.layers[layer_idx].name.lower()

#             if noise_level > 0 and ('conv2d' in layer_name or 'convlstm2d' in layer_name or 'dense' in layer_name):
#                 # Apply noise to Conv2D, ConvLSTM2D, Dense layers
#                 noise = noise_level * np.random.normal(
#                     loc=0,
#                     scale=np.std(client_params[layer_idx].ravel())
#                 )
#                 client_params[layer_idx] += noise

#                 # If the layer has BatchNormalization, apply noise to its gamma and beta parameters
# #                 if 'batchnormalization' in layer_name:
# #                     gamma_idx = layer_idx + 1  # Assuming gamma comes after beta in BatchNormalization
# #                     beta_idx = layer_idx + 2
# #                     client_params[gamma_idx] += noise
# #                     client_params[beta_idx] += noise

#         except IndexError:
#             # Handle the case where the index is out of bounds
#             print("IndexError: layer index {} is out of bounds.".format(layer_idx))
#             pass

#     client_model.set_weights(client_params)
#     return client_model


# In[12]:


def dp(client_model, noise_level):
    client_params = client_model.get_weights()

    for layer_idx in range(len(client_params)):
        if layer_idx >= len(client_model.layers):
            continue

        if 'labels' in client_model.layers[layer_idx].name:
            continue
        else:
            if noise_level > 0 and 'bias' not in client_model.layers[layer_idx].name:
                noise = noise_level * np.random.normal(
                    loc=0,
                    scale=np.std(client_params[layer_idx].ravel())
                )
                client_params[layer_idx] += noise

    client_model.set_weights(client_params)
    return client_model


# In[5]:


smlp_global = SimpleMLP()
global_model = smlp_global.build()
global_model.summary()


# In[15]:


# model1=SEInception(128,128,3,64)
# global_model=model1.SEInception_v4()

        
# global_model.compile(
#             loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy']
#         )


# In[13]:


client_names = list(clients_batched.keys())
total_samples=2201
print(client_names)


# In[ ]:


# # Federated training loop
# acc = []
# num_rounds = 20
# # noise = 0.001

# for epoch in range(num_rounds):
#     # List to store client updates
#     client_updates = []
#     global_weights = global_model.get_weights()

#     # Train each client model
#     for client in client_names:
# #         model1 = SEInception(128, 128, 3, 64)
# #         local_model = model1.SEInception_v4()
#         smlp_global = SimpleMLP()
#         local_model = smlp_global.build()
#         local_model.compile(
#             loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy'])
        

#         if client == 'client1':
#             history = local_model.fit(
#                 np.array(clients_batched[client][0]),
#                 np.array(clients_batched[client][1]),
#                 validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
#                 epochs=2,
#                 batch_size=32,
#                 verbose=2
#             )
#         elif client == 'client2':
#             history = local_model.fit(
#                 np.array(clients_batched[client][0]),
#                 np.array(clients_batched[client][1]),
#                 validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
#                 epochs=2,
#                 batch_size=32,
#                 verbose=2
#             )
#         else:
#             history = local_model.fit(
#                 np.array(clients_batched[client][0]),
#                 np.array(clients_batched[client][1]),
#                 validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
#                 epochs=2,
#                 batch_size=32,
#                 verbose=2
#             )

#         # Calculate the difference between client model and server model
#         local_model = dp(local_model, noise)
#         client_update = []
#         for i, (server_layer, client_layer) in enumerate(zip(global_model.layers, local_model.layers)):
#             if isinstance(server_layer, tf.keras.layers.BatchNormalization):
#                 # Use i to index the layers, not clients_batched[client]
#                 scale_factor = len(clients_batched[client]) / total_samples
#                 # Update the moving mean and variance of BatchNormalization layer
#                 updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
#                 updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance
#                 # Apply the updates to the server model's BatchNormalization layer
#                 server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
#             else:
#                 # For other layers, update weights directly
#                 server_layer.set_weights(client_layer.get_weights())

#             client_update.append(server_layer.get_weights())

#         client_updates.append(client_update)

#     # Aggregate client updates to update the server model
#     # Manually calculate the mean of weights
#     aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object), axis=0) for i in range(len(client_updates[0]))]
#     # Aggregate client updates to update the server model
# #     Aggregate client updates to update the server model
    


#     # Apply the aggregated update to the server model
#     for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
#         server_layer.set_weights(aggregated_weights)

#     global_acc, global_loss = test_model(test, one_hot_labels, global_model, epoch)
#     acc.append(global_acc)


# In[14]:


# Federated training loop
acc = []
num_rounds = 50
# Define the privacy budget
total_privacy_budget = 1.0

# Initialize privacy budget for each client
client_privacy_budgets = {client: total_privacy_budget / len(client_names) for client in client_names}

for epoch in range(num_rounds):
    # List to store client updates
    client_updates = []
    global_weights = global_model.get_weights()

    # Train each client model
    for client in client_names:
        smlp_global = SimpleMLP()
        local_model = smlp_global.build()
        local_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        if client == 'client1':
            history = local_model.fit(
                np.array(clients_batched[client][0]),
                np.array(clients_batched[client][1]),
                validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
                epochs=2,
                batch_size=32,
                verbose=2
            )
        elif client == 'client2':
            history = local_model.fit(
                np.array(clients_batched[client][0]),
                np.array(clients_batched[client][1]),
                validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
                epochs=2,
                batch_size=32,
                verbose=2
            )
        else:
            history = local_model.fit(
                np.array(clients_batched[client][0]),
                np.array(clients_batched[client][1]),
                validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
                epochs=2,
                batch_size=32,
                verbose=2
            )

        # Calculate the relative loss of the client
        relative_loss = history.history['loss'][-1] / total_privacy_budget

        # Calculate the privacy budget for this round
        privacy_budget = client_privacy_budgets[client] * relative_loss

        # Update privacy budget for the next round
        client_privacy_budgets[client] -= privacy_budget

        # Apply differential privacy to the local model
        local_model = dp(local_model, privacy_budget)
        client_update = []
        for i, (server_layer, client_layer) in enumerate(zip(global_model.layers, local_model.layers)):
            if isinstance(server_layer, tf.keras.layers.BatchNormalization):
                # Use i to index the layers, not clients_batched[client]
                scale_factor = len(clients_batched[client]) / total_samples
                # Update the moving mean and variance of BatchNormalization layer
                updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
                updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance
                # Apply the updates to the server model's BatchNormalization layer
                server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
            else:
                # For other layers, update weights directly
                server_layer.set_weights(client_layer.get_weights())

            client_update.append(server_layer.get_weights())

        client_updates.append(client_update)
        
    aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object), axis=0) for i in range(len(client_updates[0]))]
    # Aggregate client updates to update the server model
#     Aggregate client updates to update the server model
    


    # Apply the aggregated update to the server model
    for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
        server_layer.set_weights(aggregated_weights)

    global_acc, global_loss = test_model(test, one_hot_labels, global_model, epoch)
    acc.append(global_acc)
    

        # ... (rest of the code remains unchanged)


# In[18]:


len(client_updates[0])


# In[19]:


len(clients_batched['client3'][1]) / total_samples


# In[20]:


type(clients_batched['client3'])


# In[15]:


import matplotlib.pyplot as plt
plt.plot(acc)
# plt.ylim(.7,.9)


# In[71]:


a=np.load("iid_densenet_adaptive_DP.npy")
# b=np.load("iid_moblenetV2_adaptive_DP.npy")
plt.plot(a,label='densnet')
plt.plot(acc,label='mobilenet')
# plt.plot(acc,label='resnet')
plt.legend()


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# Assuming 'model' is your trained model and 'X_test' is your test data
y_score = global_model.predict(test)

# Binarize the labels
y_test_bin = label_binarize(one_hot_labels, classes=[0, 1, 2, 3, 4])  # Replace with your actual class labels

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8, 8))
lw = 2

# Plot each class
for i, color in zip(range(n_classes), cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

# Plot the random chance line
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[16]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Assuming you have predictions and true labels
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate Accuracy
accc = accuracy_score(y_true, y_pred)

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true, y_pred)

# Calculate Balanced Accuracy
bacc = balanced_accuracy_score(y_true, y_pred)

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multiclass

# Calculate Precision for multiclass
precision = precision_score(y_true, y_pred, average='weighted')

# Calculate Recall for multiclass
recall = recall_score(y_true, y_pred, average='weighted')



# # Calculate AUC (Area Under the Curve)
# # roc_auc = roc_auc_score(y_true, y_pred, average='weighted')

# # Create a confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)

# # Calculate Geometric Mean from the confusion matrix
# # tn, fp, fn, tp, = conf_matrix.ravel()
# # g_mean = (tp / (tp + fn)) * (tn / (tn + fp))**0.5

# # Print or use these metrics as needed
print("Accuracy:", accc)
print("Cohen's Kappa:", kappa)
print("Matthews Correlation Coefficient:", mcc)
print("Balanced Accuracy:", bacc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
# print("AUC (Area Under the Curve):", roc_auc)
# print("Geometric Mean:", g_mean)
from sklearn.metrics import roc_auc_score

# Assuming you have true labels and predicted probabilities for each class
y_true = one_hot_labels  # Replace with your true labels
y_prob = global_model.predict(test)  # Replace with your predicted probabilities

# # Calculate AUC for multiclass classification
auc = roc_auc_score(y_true, y_prob, average='weighted')

# # Print or use the AUC value as needed
print("AUC (Area Under the Curve):", auc)
from sklearn.metrics import confusion_matrix
import numpy as np

# Assuming you have true labels and predicted labels for multiclass classification
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)  # Replace with your predicted labels

# Convert true and predicted labels to class labels (not one-hot encoded)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate G-Mean for each class
class_g_means = []
for class_label in range(5):  # Replace num_classes with the number of classes
    # Create a binary confusion matrix for the current class
    true_class = (y_true == class_label)
    pred_class = (y_pred == class_label)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()

    # Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate G-Mean for the current class
    g_mean = np.sqrt(sensitivity * specificity)

    class_g_means.append(g_mean)

# Calculate the overall G-Mean (geometric mean of class G-Means)
overall_g_mean = np.prod(class_g_means) ** (1 / len(class_g_means))

# Print or use the overall G-Mean as needed
print("Overall G-Mean:", overall_g_mean)


# In[17]:


acccc=np.array(acc)
np.save("iid_resnet_adaptive_DP",acccc)


# In[18]:


global_model.save("iid_resnet_adaptive_DP.h5")


# In[26]:


import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


# In[27]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, MaxPooling2D, AveragePooling2D, concatenate, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D
def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x
def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch5x5 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch5x5 = Conv_2D_Block(branch5x5, filterB2_2, (5, 5))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_A'+str(i))

    return out

def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (7, 1))

    branch7x7dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (7, 1))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (7, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_Block_B'+str(i))

    return out
def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3_2 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3_3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 1))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (1, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 1))
    branch3x3dbl_2 = Conv_2D_Block(branch3x3dbl, filterB3_3, (1, 3))
    branch3x3dbl_3 = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3_2, branch3x3_3, branch3x3dbl_2, branch3x3dbl_3, branch_pool], axis=-1, name='Inception_Block_C'+str(i))

    return out

def Reduction_Block_A(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block A (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out

def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block B (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (1, 7))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (7, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out
def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, num_filters])(excitation)

    scale = inputs * excitation

    return scale


# In[28]:


class SEInception:
    def __init__(self, length, width, num_channel, num_filters, ratio=4, problem_type='Classification',
                 output_nums=5, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.ratio = ratio
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs


    def SEInception_v4(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)

        branch1 = Conv_2D_Block(x, 96, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        # 4x Inception-A Blocks - 35 x 256
        for i in range(4):
            x = Inception_Module_A(x, 96, 64, 96, 64, 96, 96, 96, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 192, 224, 256, 1)  # Reduction Block 1: 17 x 768

        # 7x Inception-B Blocks - 17 x 768
        for i in range(7):
            x = Inception_Module_B(x, 384, 192, 256, 192, 224, 256, 128, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 192, 256, 320, 320, 2)  # Reduction Block 2: 8 x 1280

        # 3x Inception-C Blocks: 8 x 2048
        for i in range(3):
            x = Inception_Module_C(x, 256, 384, 512, 384, 512, 512, 256, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = tf.expand_dims(x,axis = 1)
        x = ConvLSTM2D(filters=512, kernel_size=(1,1),padding = "same")(x)
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
        x = Activation('relu')(x)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v4')

        return model


# In[29]:


# class SEInceptionV3:
#     def __init__(self, length, width, num_channel, num_filters, ratio=4, problem_type='Classification', output_nums=5, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
#         self.length = length
#         self.width = width
#         self.num_channel = num_channel
#         self.num_filters = num_filters
#         self.ratio = ratio
#         self.problem_type = problem_type
#         self.output_nums = output_nums
#         self.pooling = pooling
#         self.dropout_rate = dropout_rate
#         self.auxilliary_outputs = auxilliary_outputs

#     def MLP(self, x):
#         if self.pooling == 'avg':
#             x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         elif self.pooling == 'max':
#             x = tf.keras.layers.GlobalMaxPooling2D()(x)
#         if self.dropout_rate:
#             x = tf.keras.layers.Dropout(self.dropout_rate)(x)
#         x = tf.keras.layers.Flatten()(x)
#         outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
#         if self.problem_type == 'Classification':
#             outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

#         return outputs

#     def SEInceptionV3(self):
#         inputs = tf.keras.Input((self.length, self.width, self.num_channel))

#         x = Conv_2D_Block(inputs, 32, 3, strides=2, padding='valid')
#         x = Conv_2D_Block(x, 32, 3, padding='valid')
#         x = Conv_2D_Block(x, 64, 3)

#         branch1 = Conv_2D_Block(x, 80, 3, strides=2, padding='valid')
#         branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
#         x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

#         branch1 = Conv_2D_Block(x, 192, 1)
#         branch1 = Conv_2D_Block(branch1, 192, 3, padding='valid')
#         branch2 = Conv_2D_Block(x, 192, 1)
#         branch2 = Conv_2D_Block(branch2, 192, 7)
#         branch2 = Conv_2D_Block(branch2, 192, 3, padding='valid')
#         x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

#         # Inception Blocks
#         x = Inception_Module_A(x, 64, 96, 128, 16, 32, 32, 32, 1)
#         x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

#         x = Inception_Module_B(x, 128, 128, 192, 32, 96, 64, 64, 2)
#         x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

#         x = Inception_Module_C(x, 192, 384, 384, 48, 128, 128, 128, 3)
#         x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

#         # Reduction Blocks
#         x = Reduction_Block_A(x, 192, 192, 256, 384, 256, 1)
#         x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

#         x = Reduction_Block_B(x, 256, 256, 384, 256, 256, 2)
#         x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

#         # Apply Global Average Pooling, Dropout, and Final Dense MLP Layer for the outputs
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         if self.dropout_rate:
#             x = tf.keras.layers.Dropout(self.dropout_rate)(x)

#         final_output = self.MLP(x)

#         # Create model
#         model = tf.keras.Model(inputs, final_output, name='SEInceptionV3')

#         if self.auxilliary_outputs:
#             # You may add auxilliary outputs as needed
#             aux_output_0 = self.MLP(auxiliary_output_0_tensor)
#             aux_output_1 = self.MLP(auxiliary_output_1_tensor)
#             model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='SEInceptionV3')

#         return model


# In[30]:


model1=SEInception(128,128,3,64)
global_model=model1.SEInception_v4()
global_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )


# In[31]:


# model1=SEInceptionV3(128,128,3,64)
# global_model=model1.SEInceptionV3()
# global_model.compile(
#             loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy']
#         )


# In[ ]:




