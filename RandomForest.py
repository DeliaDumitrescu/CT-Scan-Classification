from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy as np
from sklearn import preprocessing


# Train - SIMPLE RANDOM FOREST
X_train, y_train = [], []
file = open('../train.txt','r')
for line in file:
    filename, label = line.split(',')
    label = label[0]
    img = cv2.imread('../train/' + filename).flatten()
    X_train.append(img)
    y_train.append(label)
file.close()

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

RandomForest_classifier = RandomForestClassifier(n_estimators=500)
RandomForest_classifier.fit(X_train, y_train)

# Validate - SIMPLE RANDOM FOREST
file = open('../validation.txt','r')
y_true, images = [], []
for line in file:
    filename, label = line.split(',')
    img = cv2.imread('../validation/' + filename).flatten()
    images.append(img)
    y_true.append(label)
file.close()

images = scaler.transform(images)
y_true = [y[0] for y in y_true]
y_pred = RandomForest_classifier.predict(images)

# Accuracy score
print(accuracy_score(y_true, y_pred))

# Confusion matrix
matrix = confusion_matrix(y_true, y_pred)
dataframe = pandas.DataFrame(matrix / np.sum(matrix) * 10, index = ['0', '1', '2'], columns = ['0', '1', '2'])
plt.figure(figsize = (15, 10))
seaborn.heatmap(dataframe, annot=True, cmap="Purples")
plt.savefig('confusion_matrix_std_500.png')

# Test - SIMPLE RANDOM FOREST
file = open('../test.txt','r')
submission_file = open('random_forest_std_500_submission.txt', 'w')
submission_file.write('id,label\n')
images = []

for line in file:
    filename = line.split()[0]
    img = cv2.imread('../test/' + filename).flatten()
    images += [img]
    
images = scaler.transform(images)
y_pred = RandomForest_classifier.predict(images)

file.close()
file = open('../test.txt','r')
for i, line in enumerate(file):
    filename = line.split()[0]
    label = y_pred[i]
    submission_file.write(filename + ',' + label + '\n')
    
file.close()
submission_file.close()