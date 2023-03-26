import os
import pandas as pd
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.preprocessing import StandardScaler
import xgboost
# from sklearn.model_selection import cross_val_score
import numpy as np
# import matplotlib.pyplot as plt
import sklearn


# Importing dataset
dataset = pd.read_csv("parkinsons.data")

# Independant and dependant variable
X = dataset.loc[:, dataset.columns != "status"]
y = dataset["status"].values


# Y already binary variable, does not need scaling


#Get names of columns for new dataframe
names = []
for i in list(dataset):
    names.append(i)


# encode dependant variable into numerical catagorical data

normalized_items = []

le = sklearn.preprocessing.LabelEncoder()
names_col = le.fit_transform(X.loc[:, "name"])

dataset = dataset.drop("name", axis=1)

normalized_items.append(names_col)

data_list = []

# statuses_col = X.loc[:, "status"]
#
# normalized_items.append(statuses_col)



# Scale independant variable within range of 0 - 1
columns = dataset.columns
# other_names = dataset.loc[:, dataset.columns != "status"]
other_names = dataset

dataset_items = []
for (i, v) in other_names.iteritems():
    dataset_items.append(v.values)

def normalize_data(other_names):
    normalized = (other_names-np.min(other_names))/(np.max(other_names)-np.min(other_names))
    return normalized

counter = 0
for i in dataset_items:
    # print(i)
    if counter != 17:
        item = normalize_data(i)
        normalized_items.append(item)
    counter += 1

new_normalized_items = []
for i in normalized_items:
    new_normalized_items.append(i.tolist())

items_dict = {}
for i in new_normalized_items:
    items_dict[names[new_normalized_items.index(i)]] = i

revised_dataset = pd.DataFrame.from_dict(items_dict)


X = revised_dataset.loc[:, revised_dataset.columns != "status"]
y = revised_dataset["status"].values

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train)
# print(y_train)


# Training XGBoost on the Training set
classifier = xgboost.XGBClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
# y_pred = classifier.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

