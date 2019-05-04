# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# Import Dataset and split to dependent and independent variables
df = pd.read_csv('iris.csv')
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4]

# Split into train and test sets
from sklearn.model_selection import train_test_split
# Using the default size for test: 25% of total
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25)

dfTrain = pd.concat([X_Train,Y_Train],axis = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

# k-NN doesn’t perform well on imbalanced data
# Check number of members for each class in the train dataset
plt.figure(1)
sns.countplot(x='Class', data = dfTrain)

# k-NN algorithm is very sensitive to outliers
# Check the train data for outliers
i = 2
for s in df.columns.values[:4]:
    plt.figure(i)
    sns.boxplot(y=dfTrain[s], x=dfTrain['Class'])
    i += 1

plt.show()

# Fit the k-NN model to the train data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

classifier.fit(X_Train, Y_Train)

# Using k-fold cross validation to estimate model accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_Train, Y_Train, cv=5)
print('Mean accuracy = ' + str(accuracies.mean()))
print('Std = ' + str(accuracies.std()))

# Using grid search to optimize the model hyper-parameter
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':[3,5,7,8,9,11,13]}
grid_search = GridSearchCV(classifier, parameters, scoring= 'accuracy', cv=5)
grid_search.fit(X_Train,Y_Train)

print('Best params = ' + str(grid_search.best_params_))
print('With a best score: ' + str(grid_search.best_score_))

# Use the model to classify the test data
classifier = KNeighborsClassifier(n_neighbors = grid_search.best_params_['n_neighbors'])

classifier.fit(X_Train, Y_Train)
Y_Predict = classifier.predict(X_Test)

from sklearn.metrics import confusion_matrix
print('Confusion Matrix:\n' + str(confusion_matrix(Y_Test, Y_Predict)))