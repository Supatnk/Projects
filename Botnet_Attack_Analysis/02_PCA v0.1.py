import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import glob
from pathlib import Path
from pathlib import PurePath
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



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
x = df.loc[:, list(df_features)].values

# Separating out the target
y = df.loc[:, ['AttackBenignFlag']].values

#so
# Standardizing the features
x_std = StandardScaler().fit_transform(x)

# Create a PCA that will retain 75% of the variance
pca = PCA(n_components=0.75, whiten=True)

# Conduct PCA
X_pca = pca.fit_transform(x_std)
# Show results
print('Original number of features:', x_std.shape[1])
print('Reduced number of features:', X_pca.shape[1])

print(X_pca)

# pca = PCA(n_components=4)
# principalComponents = pca.fit_transform(x_std)
# principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
# finalDf = pd.concat([principalDf, df[['AttackBenignFlag']]], axis=1)

print("Break down of each feature")
print(pca.explained_variance_ratio_)
print("Break down of each feature cumulatively")
print(pca.explained_variance_ratio_.cumsum())


# to do
# could we use a random sample for PCA
# what are features
