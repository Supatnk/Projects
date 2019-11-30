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

print(df.describe())

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
#plt.show()

#After
sns.boxplot(x='num',y= 'cp',data=df_out)
#plt.show()

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
# sns.distplot(df_out['num'])
# plt.show()
#################################################################################################################################

print("Correlation")
# sns.pairplot(df_out)


plt.figure(figsize=(40,40))
sns.heatmap(df_out.corr(), annot=True, cmap="YlGnBu")
#plt.show()
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

print("PCA # First iteration -- ALL 9 FIELDS")
from sklearn.preprocessing import StandardScaler
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'oldpeak', 'slope', 'ca', 'thal']
# Separating out the features
#x = df.loc[:, features].values

#df_out(pd.to_numeric, errors='coerce').dtypes

x = df_out.loc[:, features].values

# Separating out the target
y = df_out.loc[:, ['num']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8','PC9'])

finalDf = pd.concat([principalDf, df[['num']]], axis=1)

#print(principalComponents.explained_variance_)
print("Break down of each feature")
print(pca.explained_variance_ratio_)
print("Break down of each feature cumulatively - The first 5 features make 75% ")
print(pca.explained_variance_ratio_.cumsum())


#################################################################################################################################

print("PCA # Second iteration -- First 5 fields make 75%")
from sklearn.preprocessing import StandardScaler
features = ['age', 'sex', 'cp', 'trestbps', 'chol']
# Separating out the features
#x = df.loc[:, features].values

#df_out(pd.to_numeric, errors='coerce').dtypes

x = df_out.loc[:, features].values

# Separating out the target
y = df_out.loc[:, ['num']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

finalDf = pd.concat([principalDf, df[['num']]], axis=1)

#print(principalComponents.explained_variance_)
print("Break down of each feature")
print(pca.explained_variance_ratio_)
print("Break down of each feature cumulatively - The first 3 features make 70% ")
print(pca.explained_variance_ratio_.cumsum())
#################################################################################################################################

print("PCA # Third iteration -- First 2  fields make 75%")
from sklearn.preprocessing import StandardScaler
features = ['age', 'sex', 'cp']
# Separating out the features
#x = df.loc[:, features].values

#df_out(pd.to_numeric, errors='coerce').dtypes

x = df_out.loc[:, features].values

# Separating out the target
y = df_out.loc[:, ['num']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])

finalDf = pd.concat([principalDf, df[['num']]], axis=1)

#print(principalComponents.explained_variance_)
print("Break down of each feature")
print(pca.explained_variance_ratio_)
print("Break down of each feature cumulatively - The first 2 features make 75% ")
print(pca.explained_variance_ratio_.cumsum())

#################################################################################################################################
print("KNearest neighbors Classifier model")
# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

x = df_out.loc[:, features].values
y = df_out.loc[:, ['num']].values

#kf = KFold(n_splits=10, n_repeats=5, random_state=None)
kf = KFold(n_splits=5, random_state=1, shuffle=True)
kf.get_n_splits(x)
knn = KNeighborsClassifier(n_neighbors=2)

scores = []

for train_index, test_index in kf.split(x):
	X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
	knn.fit(X_train, y_train.ravel())
	scores.append(knn.score(X_test, y_test))

# Instead of saving 10 scores in object named score and calculating mean
# We're just calculating the mean directly on the results
#print(cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy').mean())

print(scores)

print("Cross validation")
print(cross_val_score(knn, x, y.ravel(), cv=5))
print(cross_val_score(knn, x, y.ravel(), cv=5).mean())

#print("Cross validation Predict")
#y_pred= cross_val_predict(knn, x, y.ravel(), cv=5)

# print("Confusion_matrix",metrics.confusion_matrix(y_test,y_pred))
# print("ClassificationReport",metrics.classification_report(y_test,y_pred))
# print("AccuracyScore", metrics.accuracy_score(y_test, y_pred))

#################################################################################################################################
print("Support Vector Machines")
#Import svm model
from sklearn import svm

#Create a svm Classifier
kf = KFold(n_splits=5, random_state=1, shuffle=True)
kf.get_n_splits(x)
clf = svm.SVC(kernel='linear') # Linear Kernel

scores = []

for train_index, test_index in kf.split(x):
	X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
	clf.fit(X_train, y_train.ravel())
	scores.append(clf.score(X_test, y_test))

print(scores)

print("Cross validation")
print(cross_val_score(clf, x, y.ravel(), cv=5))
print(cross_val_score(clf, x, y.ravel(), cv=5).mean())

#Predict the response for test dataset
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))

#################################################################################################################################
print("Naive Bayes Classification")
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

scores = []

for train_index, test_index in kf.split(x):
	X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
	gnb.fit(X_train, y_train.ravel())
	scores.append(gnb.score(X_test, y_test))

print(scores)

print("Cross validation")
print(cross_val_score(gnb, x, y.ravel(), cv=5))
print(cross_val_score(gnb, x, y.ravel(), cv=5).mean())

# #Predict the response for test dataset
# y_pred = gnb.predict(X_test)
#
# #Predict the response for test dataset
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))
