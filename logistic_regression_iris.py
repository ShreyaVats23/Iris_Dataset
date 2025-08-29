
# **Step 1: AIM**

classification

# **Step 2: loading data**
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = sns.load_dataset('iris')
df

"""# **step 3: Data cleaning**"""

df.info()

# clean dataset
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df

df.isnull().sum()

df.max()

"""# **Step 4: Data Visualization**"""

# scatter plot

# boxplot

# regression plot

"""# **Step 5: Finding Correlation**"""

# outliers calculation - IQR

#heatmap

sns.heatmap(df.drop('species', axis=1).corr(), annot=True)

"""# **Step 6: Deciding Multi or Binary**"""

df1 = df[df['species'] != 'setosa'].copy()
df1['species'] = df1['species'].map({'versicolor': 0, 'virginica': 1})
df1

df2 = df[df['species'] != 'versicolor'].copy()
df2['species'] = df2['species'].map({'setosa': 0, 'virginica': 1})
df2

df3 = df[df['species'] != 'virginica'].copy()
df3['species'] = df3['species'].map({'versicolor': 0, 'setosa': 1})
df3

df4 = df.copy()
df4['species'] = df4['species'].map({'versicolor': 0, 'setosa': 1, 'virginica': 2})
df4

"""# **Step 7: Mapping**"""

# Multiclass classification
x = df4.iloc[:, 0:4]
y = df4.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)


classifier = LogisticRegression()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

#sample input
a = np.array([[4.6, 3.1,    1.5,    0.2 ]])
pred = classifier.predict(a)[0]
print("Predicted outcome:", pred)

from sklearn import tree

X, y = df4.iloc[:, 0:4], df4.iloc[:, 4]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)

#decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
x = df4.iloc[:, 0:4]
y = df4.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)


classifier1 = DecisionTreeClassifier()
classifier1.fit(x_train, y_train)
y_pred1 = classifier1.predict(x_test)
acc1 = accuracy_score(y_test, y_pred1)
print(acc1)

#sample input
pred1 = classifier1.predict(a)[0]
print("Predicted outcome:", pred1)

#random forest
from sklearn.ensemble import RandomForestClassifier

x = df4.iloc[:, 0:4]
y = df4.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)

classifier2 = RandomForestClassifier()
classifier2.fit(x_train, y_train)
y_pred2 = classifier2.predict(x_test)
acc2 = accuracy_score(y_test, y_pred2)
print(acc2)

#sample input
pred2 = classifier2.predict(a)[0]
print("Predicted outcome:", pred2)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)
y_pred3 = model.predict(x_test)
acc3 = accuracy_score(y_test, y_pred3)
print(acc3)

#sample input
a = np.array([[4.6,	3.1,	1.5,	0.2	]])
pred3 = model.predict(a)[0]
print("Predicted outcome:", pred3)

#SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = SVC()
model.fit(x_train, y_train)
y_pred4 = model.predict(x_test)
acc4 = accuracy_score(y_test, y_pred4)
print(acc4)

#sample input
a = np.array([[4.6,	3.1,	1.5,	0.2	]])
pred4 = model.predict(a)[0]
print("Predicted outcome:", pred4)

a = pd.DataFrame({'Algorithm': ['Multi-classification', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVM'], 'Accuracy': [acc, acc1, acc2, acc3, acc4], 'Input': [a,a,a,a, a],'Actual': [1,1,1,1,1], 'Prediction': [pred, pred1,pred2, pred3, pred4]})
a

