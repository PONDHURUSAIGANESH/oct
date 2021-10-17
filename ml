# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 22:51:26 2021

@author: Dell
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.datasets import load_iris 

iris = load_iris()

data = iris.data

data = pd.DataFrame(data,columns=iris.feature_names)
data['label']=iris.target


plt.scatter(data.iloc[:,2],data.iloc[:,3], c = iris.target)

plt.xlabel('petal length(cm)')
plt.ylabel('petal width(cm)')
plt.title('lab')
plt.legend()
plt.show()

x = data.iloc[:,0:4]

y = data.iloc[:,4]

"""
k-NN classifier
 
"""
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors = 6,metric = 'minkowski',p = 1)
kNN.fit(x,y)

xn = np.array([[5.6,3.4,5.0,0.1]])
kNN.predict(xn)


yn = np.array([[4.5,3.4,9.8,29.0]])
kNN.predict(yn)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y , train_size= 0.8 , random_state= 88,shuffle =True,stratify = y)

   
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors = 6,metric = 'minkowski',p = 1)
kNN.fit(xtrain,ytrain)

kNN.predict(xtest)
