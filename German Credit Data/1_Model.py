#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("diabetes.csv")
df.head (10)


# In[3]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[5]:


def showResult(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)

    plot_confusion_matrix(clf, X_test, y_test)  
    plt.show()
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


# In[6]:


## kNN - DT - RF


# In[7]:


estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('dt', DecisionTreeClassifier(random_state=0))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator= RandomForestClassifier(n_estimators=10, random_state=42)
)

showResult(clf, X_train, y_train, X_test, y_test)


# In[8]:


## kNN - RF - DT


# In[9]:


estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator= DecisionTreeClassifier(random_state=0)
)

showResult(clf, X_train, y_train, X_test, y_test)


# In[10]:


## DT - RF - kNN


# In[11]:


estimators = [
    ('dt', DecisionTreeClassifier(random_state=0)),
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator= KNeighborsClassifier(n_neighbors=3)
)

showResult(clf, X_train, y_train, X_test, y_test)


# In[ ]:




