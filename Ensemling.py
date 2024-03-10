#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# In[2]:


from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from classification_models.keras import Classifiers
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import tensorflow as tf
import argparse
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import datetime
import keras
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
#from keras_cv.layers import RandomCutout
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from classification_models.keras import Classifiers
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from classification_models.keras import Classifiers
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import get_file
import os,glob
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import keras
from classification_models.keras import Classifiers
import numpy as np
import cv2
import glob
import pickle
#import clahe
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
import re
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sympy.solvers import solve
from sympy import Symbol
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf   
import keras
from classification_models.keras import Classifiers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import cv2
import math



# In[ ]:





# In[3]:


from tensorflow.keras.models import load_model
model1=load_model("non-iid_densenet_adaptive_dp.h5")
model2=load_model("non-iid_resnet_adaptive_dp.h5")
model3=load_model("non-iid_MobileNetV2_adaptive_dp.h5")


# In[4]:


test=np.load("test.npy")
label=np.load("one_hot_labels.npy")
test=test/255

# one_hot_labels=label


# In[5]:





# In[6]:





# In[7]:





# In[8]:





# In[9]:


# test=test/255


# In[10]:





# In[11]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


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





# In[13]:


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


# In[15]:


acc_dense


# In[16]:


correct=0
pred_resnet=model2.predict(test)
pred_resnet = np.argmax(pred_resnet, axis=1)
for i in range(len(label)):
                
                if label[i]==pred_resnet[i]: 
                    
                   
                    correct+=1
acc_resnet=(correct/len(label))
acc_resnet


# In[17]:


correct=0
pred_inception=model3.predict(test)
pred_inception = np.argmax(pred_inception, axis=1)
for i in range(len(label)):
                
                if label[i]==pred_inception[i]: 
                    
                   
                    correct+=1
acc_inception=(correct/len(label))
acc_inception


# In[18]:


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


# In[19]:


one_hot_matrix.shape


# In[20]:


# test=test/255


# In[21]:


# model=load_model("fl_non-IID_DenseNet201_50.h5")
# correct=0
# # pred_inception=model.predict(test)
# # print(pred_inception)
# loss,accuracy=model.evaluate(test2,label2)
# accuracy


# In[22]:


pred_dense=model1.predict(test)
pred_resnet=model2.predict(test)


pred_inception=model3.predict(test)


# In[23]:


pred_dense


# In[24]:


# pred_resnet


# In[25]:




# import numpy as np

# # Your integer labels


# # Find the number of unique labels
# num_classes = len(np.unique(pred_resnet))

# # Initialize an empty one-hot encoding matrix with zeros
# pred_resnet = np.zeros((len(pred_resnet), num_classes))

# # Fill the matrix with 1s in the appropriate columns
# for i, label in enumerate(pred_resnet):
#     pred_resnet[i, label] = 1

# Print the one-hot encoded matrix
# print(one_hot_matrix)


# In[26]:


pred_dense


# In[27]:


weighted_average(one_hot_matrix,pred_dense, acc_dense, pred_resnet, acc_resnet, pred_inception,acc_inception)
majority_voting(one_hot_matrix,pred_dense, acc_dense, pred_resnet, acc_resnet, pred_inception,acc_inception)
averaging(one_hot_matrix,pred_dense, acc_dense, pred_resnet, acc_resnet, pred_inception,acc_inception)

            


# In[ ]:





# In[ ]:




