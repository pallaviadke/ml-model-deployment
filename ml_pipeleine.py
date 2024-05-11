# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:41:33 2024

@author: Pallavi Adke
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv('Online Purchase.csv')
training_data.describe()
X=training_data.iloc[:,:-1].values
y=training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print (classification_report(y_test, y_pred))

new_prediction = classifier.predict(sc.transform(np.array([[40, 2000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40, 2000]])))[:,1]



import pickle
model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))







