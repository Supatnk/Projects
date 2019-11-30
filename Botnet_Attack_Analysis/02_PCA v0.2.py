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

# Splitting the data set into Training and Test
x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Standard scalar normalization to normalize feature set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Create a PCA that will retain 75% of the variance
pca = PCA(n_components=0.90, whiten=True)

# Conduct PCA on Train and test data set
# sklearn fit calculates the parameters and saves them as an internal object
# transform() is then called to apply the transformation to a particular set of examples
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Show results
print('Original number of features:', x_train.shape[1])
print('Reduced number of features:', x_test_pca.shape[1])

print(x_test_pca)

# pca = PCA(n_components=4)
# principalComponents = pca.fit_transform(x_std)
# principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
# finalDf = pd.concat([principalDf, df[['AttackBenignFlag']]], axis=1)

print("Break down of each feature")
print(pca.explained_variance_ratio_)
print("Break down of each feature cumulatively")
print(pca.explained_variance_ratio_.cumsum())


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