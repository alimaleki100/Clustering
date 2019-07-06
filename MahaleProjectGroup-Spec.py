# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:40:01 2019

@author: session1
"""
############################################Libraries###########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


############################################ReadingData###########################################################
df=pd.read_excel('Dataset3-2.xls')

############################################Cleaning###########################################################
df=df.fillna(value=0)

# review dataframe
print(df.columns)




############################################DefineTargetData###########################################################
x=df.iloc[:,5:].values
x=pd.DataFrame(x)


############################################FeatureScaling###########################################################from sklearn.preprocessing import StandardScaler


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

x=pd.DataFrame(x)

############################################ExtractGroups###########################################################
x['E']=x[15]+x[16]+x[17]+x[18]+x[19]+x[20]+x[21]+x[22]+x[23]+x[24]
x['N']=x[25]+x[26]+x[27]+x[28]+x[29]+x[30]+x[31]+x[32]
x['K']=x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]
x['A']=x[33]+x[34]+x[35]

#Set cluster data
print(x.columns)

x=x.drop(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
print(x.columns)



import scipy.cluster.hierarchy as shc


############################################encoding###########################################################
"""
# Encoding the Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [9])
x = onehotencoder.fit_transform(x).toarray()

onehotencoder = OneHotEncoder(categorical_features = [17])
x = onehotencoder.fit_transform(x).toarray()
"""
############################################ClusterModeling###########################################################

from sklearn.cluster import SpectralClustering

spec = SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=0)
spec_predict=spec.fit_predict(x)

print(spec_predict)
from sklearn import metrics

from sklearn.metrics import pairwise_distances
print("Silhouette Score: %0.3f"% metrics.silhouette_score(x, spec_predict, metric='euclidean'))
print("Calinski-Harabaz Index: %0.3f"% metrics.calinski_harabaz_score(x, spec_predict))
 

############################################SavingInXlsx###########################################################

x=pd.DataFrame(x)
# Add cluster column to initial DF
predictdf=pd.DataFrame(spec_predict)
predictdf.columns=['PredictedCluster']


# Save Predict Cluster with Features

resultFeatures = pd.concat([df, predictdf], axis=1, sort=False)
writer=pd.ExcelWriter("resultSPECFG.xlsx",engine='xlsxwriter')
resultFeatures.to_excel(writer, sheet_name='new')
writer.close()

# Save Predict Cluster with Meyars
resultMeyars = pd.concat([x, predictdf], axis=1, sort=False)
writer=pd.ExcelWriter("resultSPECGG.xlsx",engine='xlsxwriter')
resultMeyars.to_excel(writer, sheet_name='new')
writer.close()

ax2 = resultMeyars.plot.scatter(x='K', y='E', c='PredictedCluster',colormap='viridis',figsize=(10,10))
ax2 = resultMeyars.plot.scatter(x='A', y='E', c='PredictedCluster',colormap='viridis',figsize=(10,10))
ax2 = resultMeyars.plot.scatter(x='N', y='E', c='PredictedCluster',colormap='viridis',figsize=(10,10))




import seaborn as sns; sns.set(color_codes=True)