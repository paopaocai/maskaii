#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay


# In[2]:


width = 416 
height = 416
array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, (width, height))
        array_of_img.append(resized_image)
        #print(img)
        #print(array_of_img)

read_directory('with_mask')

array_of_img = np.array(array_of_img)
array_of_img = array_of_img.reshape(np.shape(array_of_img)[0],np.shape(array_of_img)[1]*np.shape(array_of_img)[2]*np.shape(array_of_img)[3])
print(np.shape(array_of_img))


# In[3]:


array_of_img2 = [] # this if for store all of the image data
# this function is for read image,the input is directory name

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, (width, height))
        array_of_img2.append(resized_image)
        #print(img)
        #print(array_of_img)

read_directory('without_mask')

array_of_img2 = np.array(array_of_img2)
array_of_img2 = array_of_img2.reshape(np.shape(array_of_img2)[0]
                                                  ,np.shape(array_of_img2)[1]
                                                  *np.shape(array_of_img2)[2]
                                                  *np.shape(array_of_img2)[3])
print(np.shape(array_of_img2))


# In[4]:


array_of_img3 = [] # this if for store all of the image data
# this function is for read image,the input is directory name

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, (width, height))
        array_of_img3.append(resized_image)
        #print(img)
        #print(array_of_img)

read_directory('noseout')

array_of_img3 = np.array(array_of_img3)
array_of_img3 = array_of_img3.reshape(np.shape(array_of_img3)[0]
                                                  ,np.shape(array_of_img3)[1]
                                                  *np.shape(array_of_img3)[2]
                                                  *np.shape(array_of_img3)[3])
print(np.shape(array_of_img3))


# In[5]:


y_mask = np.zeros(np.shape(array_of_img)[0])

y_nomask = np.ones(np.shape(array_of_img2)[0])
y_nose = np.ones(np.shape(array_of_img3)[0])+1


# In[6]:


X = np.vstack((array_of_img,array_of_img2))
y = y_mask

for i in y_nomask:
    y = np.append(y, i)

X = np.vstack((X,array_of_img3))
for j in y_nose:
    y = np.append(y, j)
    


# In[20]:


from sklearn.preprocessing import normalize
X = normalize(X)
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,test_size=0.4)


# In[21]:


model1 = LogisticRegression(multi_class = 'multinomial')
model2 = RandomForestClassifier(max_depth=2, random_state=0)

models = [model1, model2]

for model in models:
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    
    disp = ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
    test_acc = np.sum(test_preds==y_test)/len(y_test)
    print(model)
    print('Test set accuracy is {:.3f}'.format(test_acc))


# In[26]:


import matplotlib.pyplot as plt
x = [ 0.12499998,  0.24999997,  0.37499995,  0.49999994,  0.62499992,
        0.74999991,  0.87499989,  0.99999988,  0.99999988,  0.99999988,
        0.99999988,  0.99999988,  0.99999988,  0.99999988,  0.99999988,
        0.99999988,  0.99999988,  0.99999988,  0.99999988,  0.99999988]
y = [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  0.88888889,  0.8       ,
        0.72727273,  0.66666667,  0.61538462,  0.57142857,  0.53333333,
        0.5       ,  0.47058824,  0.44444444,  0.42105263,  0.4       ]
plt.plot(x,y)
plt.xlabel("recall")
plt.ylabel("precision")
plt.show()


# In[ ]:




