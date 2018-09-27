"""
Predicting Survival in the Titanic Data Set
We will be using a decision tree to make predictions about the Titanic data set from
Kaggle. This data set provides information on the Titanic passengers and can be used to
predict whether a passenger survived or not.

"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree

#loading data
url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

# we use Pclass,Sex,Age,SibSp,Parch,Fare columns to build model
X = titanic[['Pclass','Age','Sex','SibSp','Parch','Fare']]
y = titanic['Survived']

#catagorizing Fare column
X['Fare'] = X['Fare'].astype(int)
X.loc[ X['Fare'] <= 7.91, 'Fare'] = 0
X.loc[(X['Fare'] > 7.91) & (X['Fare'] <= 14.454), 'Fare'] = 1
X.loc[(X['Fare'] > 14.454) & (X['Fare'] <= 31), 'Fare']   = 2
X.loc[(X['Fare'] > 31) & (X['Fare'] <= 99), 'Fare']   = 3
X.loc[(X['Fare'] > 99) & (X['Fare'] <= 250), 'Fare']   = 4
X.loc[ X['Fare'] > 250, 'Fare'] = 5
X['Fare'] = X['Fare'].astype(int)

#adding new column Fare_per_person
X['Fare_Per_Person'] = X['Fare']/(X['SibSp']+X['Parch']+1)
X['Fare_Per_Person'] = X['Fare_Per_Person'].astype(int)

#catagorizing the Age column
X['Age_new']=1
X['Age_new'][X['Age']<16] = 0
X['Age_new'][X['Age']>60] = 2
X['Age_new']=X['Age_new'].astype(int)

#catagorizing the Sex column
X['Gender'] = 0
X['Gender'][X['Sex']=="male"] = 1


X.drop(['Sex','Age'],axis=1,inplace=True)

# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)

#printing classification report
print("classification report: ")
print(classification_report(y_test, y_pred))