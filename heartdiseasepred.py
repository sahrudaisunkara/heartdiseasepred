import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import os
print(os.listdir())
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#importing the dataset
url='https://raw.githubusercontent.com/VenkateshBH99/Hybrid-Random-Forest-Linear-Model/master/Proposed_Model(%20HRFLM)/cleve.csv'
import pandas as pd
heart_data=pd.read_csv(url)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # Import the imputer

X = heart_data.drop('num', axis=1)
y = heart_data['num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize the Random Forest Classifier
randfor_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

randfor_classifier.fit(X_train, y_train)
y_pred = randfor_classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import precision_score, recall_score
# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate F-score
f_score = 2 * (precision * recall) / (precision + recall)
print(f_score)

# Calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)