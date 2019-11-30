import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print
from pyod.utils.data import get_outliers_inliers
from pyod.utils.example import visualize

#######################################################################################################################
# PCA
# do feature reduction
# find key attributes

#######################################################################################################################
# Read the combined file

filename = r'C:\Users\surya\Documents\MS Project\Botnet Attack Dataset_Outputfile\All_Devices.csv'

# Capture the features
df = pd.read_csv(filename,  index_col=None, header=0)

# Drop categorical features
df_features = df.drop(['Device', 'AttackBenign', 'AttackBenignFlag', 'AttackType'], axis=1)
list(df_features)

# Separating out the features
x = df.loc[:, list(df_features)]

# Separating out the target
y = df.loc[:, ['AttackBenignFlag']]

# Splitting the data set into Training and Test
x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Standard scalar normalization to normalize feature set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


data_scaled_train = pd.DataFrame(x_train, columns=list(df_features))
data_scaled_test = pd.DataFrame(x_test, columns=list(df_features))

print(pd.DataFrame(data_scaled_test).head(5))

# Create a PCA that will retain 75% of the variance
# pca = PCA(n_components=0.80, whiten=True)
pca = PCA(n_components=6, whiten=True)


# Conduct PCA on Train and test data set
# sklearn fit calculates the parameters and saves them as an internal object
# transform() is then called to apply the transformation to a particular set of examples
x_train_pca = pca.fit_transform(data_scaled_train)
x_test_pca = pca.transform(data_scaled_test)

# Show results
print('Original number of features:', data_scaled_train.shape[1])
print('Reduced number of features:', x_test_pca.shape[1])
print('x_test_pca shape:', x_test_pca.shape)

print("Break down of each feature")
print(pca.explained_variance_ratio_)
print("Break down of each feature cumulatively")
print(pca.explained_variance_ratio_.cumsum())

# Dump components relations with features:
#print(pd.DataFrame(pca.components_, columns=data_scaled_test.columns, index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']))

# plot the variance
# Notice that the variance kind of flats out after 10 PC's
# Hence go back and change the retention percentage to .90
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.xlabel('Cumulative explained variance')

# what are the features
# build the next models
# isolation forest
# clustering - k nearest
# auto encoder
# local Outlier factor

######################################
# K Nearest Neighbors
######################################

# contamination = 0.1  # percentage of outliers
# n_train = 200  # number of training points
# n_test = 100  # number of testing points

# train kNN detector
clf_name = 'KNN'
clf = KNN()
clf.fit(x_train_pca)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(x_test_pca)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(x_test_pca)  # outlier scores
print(y_test_scores)
# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

print("Confusion_matrix", metrics.confusion_matrix(y_test, y_test_pred))
print("ClassificationReport", metrics.classification_report(y_test, y_test_pred))
print("AccuracyScore", metrics.accuracy_score(y_test, y_test_pred))


# Internal method to separate inliers from outliers.
get_outliers_inliers(x_test_pca, y_test)
plt.scatter(X_outliers ,X_inliers )

# visualize the results
visualize(clf_name, x_train_pca, y_train, x_test_pca, y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=False)

# LOF - Local outlier factor
# https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.lof
# pyod.models.lof.LOF(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=0.1, n_jobs=-1)


# ABOD - Angle Based Outlier Detector



# Isolation Forest

