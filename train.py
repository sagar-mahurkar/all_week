# import important modules
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from zoneinfo import ZoneInfo
from datetime import datetime
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# load the data
data = pd.read_csv('data.csv')

# split the data into training and test data
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# split the train data into training and validation
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 42)

# train the model
model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
model.fit(X_train,y_train)

# dump the model
joblib.dump(model, f"artifacts/model.joblib")

# predict using the model
prediction=model.predict(X_eval)
print('Test set accuracy: ',"{:.3f}".format(metrics.accuracy_score(prediction,y_eval)))

prediction=model.predict(X_test)
print('Eval set accuracy: ',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))