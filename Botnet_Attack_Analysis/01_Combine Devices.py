import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import glob
from pathlib import Path
from pathlib import PurePath

#################################################################################################################################
# Read and Concatenate all the csv files from the specific location to a single file.

filepath = r'C:\Users\surya\Documents\MS Project\Botnet Attack Dataset - Ecobee_Thermostat'
dfall =[]
for filename in Path(filepath).glob('**/*.csv'):
    df = pd.read_csv(filename, index_col=None, header=0)
    df['Device'] = PurePath(filename).parts[6]
    df['AttackBenign'] = PurePath(filename).parts[7].replace(".csv", "")
    if 'attacks' in PurePath(filename).parts[7]:
        df['AttackBenignFlag'] = 1
    else:
        df['AttackBenignFlag'] = 0
    if PurePath(filename).parts[7] != "benign_traffic.csv":
        df['AttackType'] = PurePath(filename).parts[8].replace(".csv", "")
    else:
        df['AttackType'] = ""
    dfall.append(df)
df_final = pd.concat(dfall)

df_final.apply(pd.to_numeric, errors='coerce').dtypes

#################################################################################################################################

#Categorise dataset
df_final["Device"].astype('category')
df_final["AttackBenign"].astype('category')
df_final["AttackBenignFlag"].astype('category')
df_final["AttackType"].astype('category')

#################################################################################################################################

# Data quality checks

# 1 - No missing or null data
print("Check for Missing or Null Data points")
print(df_final.isnull().sum())

# 2 Describe dataset
# df_Describe = df_final.describe()
# print(df_Describe)
# export_Describe = df_Describe.to_csv(r'C:\Users\surya\Documents\MS Project\Botnet Attack Dataset_Outputfile\Describe.csv', index=None, header=True)

# 3 Select duplicate rows except first occurrence based on all columns
# Duplicates expected as the data is packet information , it may not be unique
# duplicateRowsDF = df_final[df_final.duplicated()]
# print("Duplicate Rows except first occurrence based on all columns are :")
# print(duplicateRowsDF)

export_csv = df_final.to_csv(r'C:\Users\surya\Documents\MS Project\Botnet Attack Dataset_Outputfile\Ecobee_Thermostat.csv', index=None, header=True)

#################################################################################################################################

# print("Outlier detection using IQR")
#
# Q1 = df_final.quantile(0.25)
# Q3 = df_final.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)

#print((df_final < (Q1 - 1.5 * IQR)) | (df_final > (Q3 + 1.5 * IQR)))

# print("after removing outliers")
# df_out = df_final[~((df_final < (Q1 - 1.5 * IQR)) | (df_final > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(df_out.shape)
#################################################################################################################################


# print("Correlation")
# # sns.pairplot(df_out)
#
# plt.figure(figsize=(40,40))
# sns.heatmap(df_final.corr(), annot=True, cmap="YlGnBu")
# plt.show()
#################################################################################################################################

#PLAN
# Understand the data
# understand the attributes
# Do PCA
# do feature reduction
# find key attributes
