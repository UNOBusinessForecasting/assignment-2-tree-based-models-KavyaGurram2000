#Decision Tree Classifier

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score


train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

X_train = train_data.drop(columns=['meal', 'id', 'DateTime'], axis=1)
y_train = train_data['meal']

#Spliting the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42, max_depth=None)
modelFit = model.fit(X_train_split, y_train_split)

#Make predictions on the validation set
val_predictions = modelFit.predict(X_val_split)

#Calculate validation accuracy
validation_accuracy = accuracy_score(y_val_split, val_predictions)
print(f"Validation Accuracy: {validation_accuracy}")


joblib.dump(modelFit, 'modelFit.pkl')

#Load the test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

test_data = test_data.drop(columns=['id', 'DateTime', 'meal'], axis=1)

# Reorder the test_data columns to match X_train
X_test = test_data[X_train.columns]

pred = modelFit.predict(X_test)
