import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics

from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.combination import aom
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.lscp import LSCP
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.xgbod import XGBOD


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
x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Standard scalar normalization to normalize feature set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

data_scaled_train = pd.DataFrame(x_train, columns=list(df_features))
data_scaled_test = pd.DataFrame(x_test, columns=list(df_features))

# print(pd.DataFrame(data_scaled_test).head(5))

classifiers = {
        'K Nearest Neighbors (KNN)': KNN(),
        'Average KNN': KNN(method='mean'),
        'Isolation Forest': IForest(),
        'Cluster-based Local Outlier Factor': CBLOF(),
        'Feature Bagging': FeatureBagging(),
        'Stochastic Outlier Selection': SOS(),
        'Local Correlation Integral': LOCI()
}

# classifiers = {'Angle-based Outlier Detector': ABOD()}

for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(x_train)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(x_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(x_test)  # outlier scores
        print(y_test_scores)
        # evaluate and print the results
        print("\nOn Training Data:")
        evaluate_print(clf_name, y_train, y_train_scores)
        print("\nOn Test Data:")
        evaluate_print(clf_name, y_test, y_test_scores)

        print("Confusion_matrix", metrics.confusion_matrix(y_test, y_test_pred))
        print("ClassificationReport", metrics.classification_report(y_test, y_test_pred))
        print("AccuracyScore", metrics.accuracy_score(y_test, y_test_pred))