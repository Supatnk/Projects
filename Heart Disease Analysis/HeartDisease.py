import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import researchpy as rp

filepath = r'C:\Users\surya\Desktop\Heart Disease Dataset\HeartDisease_Cleveland.csv'
df = pd.read_csv(filepath)
df.apply(pd.to_numeric, errors='coerce').dtypes

#df.apply(pd.to_numeric, errors='coerce').dtypes

df["age"].astype('category')
df["num"].astype('category')

print("Heart Disease data set dimensions : {}".format(df.shape))

print("Check for Missing or Null Data points")
print(df.isnull().sum())

# from bokeh.plotting import output_notebook,figure,show
# group = df.groupby('age')
# group.describe()

#################################################################################################################################

#Annotations
from bokeh.plotting import output_notebook,figure,show
from bokeh.models import ColumnDataSource, LabelSet


source = ColumnDataSource(data=df)

p = figure(title='Heart Disease Analysis',height=400,width=300,x_axis_label='Age',y_axis_label='Num')

p.circle(x='age',y='num',source=source)

label = LabelSet(x="age",y="num",text="model",source=source,x_offset=2,y_offset=2)
p.add_layout(label)
show(p)

#################################################################################################################################

print("Outlier detection using zscore")
from scipy.stats import zscore
print(df.apply(zscore))

print("define a threshold to identify an outlier")
threshold = 3
print(np.where(df > 3))

#################################################################################################################################

print("Outlier detection using IQR")

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

print("after removing outliers")
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_out.shape)


#using plot to identify any outliers
#Before
sns.boxplot(x='num',y= 'cp',data=df)
plt.show()

#After
sns.boxplot(x='num',y= 'cp',data=df_out)
plt.show()

#################################################################################################################################

# Use seaborn to check for normality distribution
# sns.distplot(df_out['age'])
# plt.show()
# sns.distplot(df_out['sex'])
# plt.show()
# sns.distplot(df_out['cp'])
# plt.show()
# sns.distplot(df_out['trestbps'])
# plt.show()
# sns.distplot(df_out['chol'])
# plt.show()
# sns.distplot(df_out['restecg'])
# plt.show()
# sns.distplot(df_out['thalach'])
# plt.show()
# sns.distplot(df_out['exang'])
# plt.show()
# sns.distplot(df_out['oldpeak'])
# plt.show()
# sns.distplot(df_out['slope'])
# plt.show()
# sns.distplot(df_out['ca'])
# plt.show()
# sns.distplot(df_out['thal'])
# plt.show()

#################################################################################################################################

#hypothesis testing chi-square test for independence
#The H0 (Null Hypothesis): There is no relationship between variable one and variable two.
#The H1 (Alternative Hypothesis): There is a relationship between variable 1 and variable 2.

#https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
#https://pythonfordatascience.org/chi-square-test-of-independence-python/

#Chi-square Test of Independence using scipy.stats.chi2_contingency
from scipy.stats import chi2_contingency
from scipy.stats import chi2

#crosstab = pd.crosstab(df['age'], df['num'])
crosstab = pd.crosstab(df_out['age'], df_out['num'])
print(crosstab)

stat, p, dof, expected = chi2_contingency(crosstab)

print('dof=%d' % dof)
print(expected)

# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

#################################################################################################################################

#Chi-square Test of Independence using Researchpy

#table, results = rp.crosstab(df_out['age'], df_out['num'], prop='col', test='chi-square')
#print(table)

#print(results)

#################################################################################################################################
#pca


# plt.scatter(df_out['age'], df_out['num'])
# plt.axis('equal')
# plt.show()

print("Principal Component Analysis")
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

filepath = r'C:\Users\surya\Desktop\Heart Disease Dataset\HeartDisease_Cleveland.csv'
data = pd.read_csv(filepath)
data.apply(pd.to_numeric, errors='coerce').dtypes

#convert it to numpy arrays
X = data.values

#pca = PCA(n_components=4)
pca = PCA(n_components=4)
pca.fit(X)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)

plt.show(var1)

X1=pca.fit_transform(X)

print(X1)

#################################################################################################################################
#PCA 2nd version # First iteration
from sklearn.preprocessing import StandardScaler
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'oldpeak', 'slope', 'ca', 'thal']
# Separating out the features
#x = df.loc[:, features].values

#df_out(pd.to_numeric, errors='coerce').dtypes

# x = df.loc[:, features].values
#
# # Separating out the target
# y = df.loc[:, ['num']].values
#
# # Standardizing the features
# x = StandardScaler().fit_transform(x)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=9)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8','PC9'])
#
# finalDf = pd.concat([principalDf, df[['num']]], axis=1)
#
# #print(principalComponents.explained_variance_)
# print(pca.explained_variance_ratio_)
# #print(pca.explained_variance_ratio_.cumsum())


#################################################################################################################################

print("PCA 2nd version # First iteration")
from sklearn.preprocessing import StandardScaler
features = ['age', 'sex', 'cp']
# Separating out the features
#x = df.loc[:, features].values

#df_out(pd.to_numeric, errors='coerce').dtypes

x = df.loc[:, features].values

# Separating out the target
y = df.loc[:, ['num']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])

finalDf = pd.concat([principalDf, df[['num']]], axis=1)

#print(principalComponents.explained_variance_)
print(pca.explained_variance_ratio_)
#print(pca.explained_variance_ratio_.cumsum())


#################################################################################################################################
print("Correlation")
# sns.pairplot(df_out)


plt.figure(figsize=(40,40))
sns.heatmap(df_out.corr(), annot=True, cmap="YlGnBu")
plt.show()
#################################################################################################################################

# from sklearn.cluster import KMeans

#
#
# kmeans = KMeans(n_clusters=2, random_state=0)
# clusters = kmeans.fit_predict(x)
# kmeans.cluster_centers_.shape
# plt.show()
#################################################################################################################################
# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

# x = df.loc[:, features].values
# y = df.loc[:, ['num']].values

x = df_out.loc[:, features].values
y = df_out.loc[:, ['num']].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test

print("KNearest neighbors Classifier model")
knn = KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

#################################################################################################################################
print("Support Vector Machines")
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

#################################################################################################################################
print("Naive Bayes Classification")
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
#################################################################################################################################

