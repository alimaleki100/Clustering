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
import seaborn



############################################ReadingData###########################################################
df=pd.read_excel('Dataset3-2.xls')

############################################Cleaning###########################################################
df=df.fillna(value=0)

# review dataframe
print(df.columns)




############################################DefineTargetData###########################################################
x=df.iloc[:,5:].values
x=pd.DataFrame(x)


xforest=df.iloc[:,5:]
xColumns=pd.DataFrame(xforest.columns)

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

x=x.drop(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
print(x.columns)

seaborn.distplot(x['A'], bins=20,color='blue')




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

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import Birch

brc = Birch(branching_factor=8, n_clusters=3, threshold=0.5,compute_labels=True)
brc_predict=brc.fit_predict(x)
print(brc_predict)

from sklearn import metrics
print("Silhouette Score: %0.3f"% metrics.silhouette_score(x, brc_predict, metric='euclidean'))
print("Calinski-Harabasz Index: %0.3f"% metrics.calinski_harabaz_score(x, brc_predict))



############################################SavingInXlsx###########################################################

x=pd.DataFrame(x)
# Add cluster column to initial DF
predictdf=pd.DataFrame(brc_predict)
predictdf.columns=['PredictedCluster']


# Save Predict Cluster with Features

resultFeatures = pd.concat([df, predictdf], axis=1, sort=False)
writer=pd.ExcelWriter("resultHCFG.xlsx",engine='xlsxwriter')
resultFeatures.to_excel(writer, sheet_name='new')
writer.close()

# Save Predict Cluster with Meyars
resultMeyars = pd.concat([x, predictdf], axis=1, sort=False)
writer=pd.ExcelWriter("resultHCGG.xlsx",engine='xlsxwriter')
resultMeyars.to_excel(writer, sheet_name='new')
writer.close()

ax2 = resultMeyars.plot.scatter(x='K', y='E', c='PredictedCluster',colormap='viridis',figsize=(10,10))

###############################Feature Importance###################################
####Feature Importance sklearn.feature_selection
"""
from sklearn.feature_selection import SelectPercentile, f_classif

selector = SelectPercentile(f_classif, percentile=20)
selector.fit(xforest, ac_predict)
scores = -np.log10(selector.pvalues_)

scoresdf=pd.DataFrame(scores)
scoresdf.columns=['FeatureImportance']

scoresResult = pd.concat([xColumns, scoresdf], axis=1)
scoresResult=scoresResult.sort_values("FeatureImportance",ascending = False)
print(scoresResult)
writer=pd.ExcelWriter("ScoresFeatureImportance.xlsx",engine='xlsxwriter')
scoresResult.to_excel(writer, sheet_name='new')
writer.close()
"""


####Feature Importance Forest
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(xforest,brc_predict)
importances = forest.feature_importances_
print(importances)


importancedf=pd.DataFrame(importances)
importancedf.columns=['FeatureImportance']
importanceResult = pd.concat([xColumns, importancedf], axis=1)
importanceResult=importanceResult.sort_values("FeatureImportance",ascending = False)
print(importanceResult)

writer=pd.ExcelWriter("ForestFeatureImportance.xlsx",engine='xlsxwriter')
importanceResult.to_excel(writer, sheet_name='new')
writer.close()






############################################Mantaghe###########################################################



from scipy.cluster import hierarchy


xMantaghe = pd.concat([x, df['Mantaghe']], axis=1, sort=False)
heatMapMantaghe=xMantaghe
xMantaghe = pd.concat([xMantaghe, predictdf], axis=1, sort=False)
xMantaghe = pd.concat([xMantaghe, df['Mahale']], axis=1, sort=False)
print(xMantaghe)


mantaghe=20

resultMantaghe=xMantaghe[xMantaghe['Mantaghe']==mantaghe]
print(resultMantaghe)
resultMantaghe=resultMantaghe.set_index('Mahale')
# Calculate the distance between each sample
Z = hierarchy.linkage(resultMantaghe, 'ward')

# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=resultMantaghe.index)

resultheatMapMantaghe=heatMapMantaghe[heatMapMantaghe['Mantaghe']==mantaghe]
resultheatMapMantaghe=resultheatMapMantaghe.drop(['Mantaghe'], axis=1)
import seaborn as sns; sns.set(color_codes=True)
g = sns.clustermap(resultheatMapMantaghe, metric="correlation",method="single",cmap="mako",figsize=(20, 10))

resultMantagheArray = np.asarray(resultMantaghe)



import seaborn as sns; sns.set()
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x=2, y=3, hue='PredictedCluster',data=resultMantaghe)



# Without regression fit:
sns.regplot(x=resultMantaghe['E'], y=resultMantaghe['N'], fit_reg=False)
plt.show()

# without regression
sns.pairplot(resultMantaghe, kind="scatter")
plt.show()


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
 

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(resultMantaghe['E'], resultMantaghe['K'], resultMantaghe['A'], 
           c=resultMantaghe['PredictedCluster'], s=70,cmap='viridis')
ax.view_init(30, 30)
plt.show()


# plot Polar
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.scatter(resultMantaghe['A'], resultMantaghe['K'], 
           c=resultMantaghe['PredictedCluster'], s=70,cmap='hsv')
plt.show()



sns.pairplot(resultMantaghe, kind="scatter",hue='PredictedCluster')
plt.show()



from matplotlib import pyplot as plt
import seaborn as sns    




from matplotlib import pyplot


fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(resultMantaghe['E'], resultMantaghe['N'], resultMantaghe['K'],
           c=resultMantaghe['PredictedCluster'], s=70,cmap='viridis')
pyplot.show()



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = resultMantaghe['E']
ys = resultMantaghe['N']
zs = resultMantaghe['K']
ax.scatter(xs, ys, zs,c='PredictedCluster', s=100, alpha=0.9, edgecolors='r')
ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Fixed Acidity')
ax.set_zlabel('Alcohol')
plt.show()



jp = sns.pairplot(data=resultMantaghe, 
                  x_vars=[0], 
                  y_vars=[1], 
                  z_vars=[2],
                  size=4.5,
                  hue="PredictedCluster", # <== 😀 Look here!
                  plot_kws=dict(edgecolor="k", linewidth=0.5))
##############

############################################Testing###########################################################




from scipy.cluster import hierarchy


# Calculate the distance between each sample
Z = hierarchy.linkage(resultMeyars, 'ward')

# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=resultMeyars.index)


import seaborn as sns; sns.set(color_codes=True)
g = sns.clustermap(x, metric="correlation",method="single",cmap="hsv",figsize=(20, 10))

resultMeyarArray = np.asarray(resultMeyars)



import seaborn as sns; sns.set()
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x=resultMeyars['K'], y=resultMeyars['E'], hue=resultMeyars['PredictedCluster'],cmap="hsv")
ax = sns.scatterplot(x=resultMeyars['A'], y=resultMeyars['N'], hue=resultMeyars['PredictedCluster'],cmap="viridis")
ax = sns.scatterplot(x=resultMeyars['K'], y=resultMeyars['A'], hue=resultMeyars['PredictedCluster'],cmap="hsv")




# Without regression fit:
#sns.regplot(x=resultMeyars['E'], y=resultMeyars['N'], fit_reg=False,color=resultMeyars['PredictedCluster'])

# without regression
sns.pairplot(resultMeyars, kind="scatter",palette="husl",hue='PredictedCluster',vars=['K','E','A','N'],height=3)
sns.pairplot(resultMeyars, kind="reg",palette="husl",vars=['K','E','A','N'])

plt.show()


sns.clustermap(x.corr(), center=0, cmap="vlag",linewidths=.75, figsize=(13, 13))


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
clusters=[0,1,2]

# plot
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Emkanat')
ax.set_ylabel('Kalbadi')
ax.set_zlabel('Amalkardi')
ax.scatter(resultMeyars['E'], resultMeyars['K'], resultMeyars['A'], 
           c=resultMeyars['PredictedCluster'], s=70,cmap='viridis')
ax.view_init(30, 30)
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()


# plot
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Emkanat')
ax.set_ylabel('Kalbadi')
ax.set_zlabel('Niroo Ensani')
ax.scatter(resultMeyars['E'], resultMeyars['K'], resultMeyars['N'], 
           c=resultMeyars['PredictedCluster'], s=70,cmap='viridis')
ax.view_init(30, 30)
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()


# plot Polar
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.scatter(resultMeyars['N'], resultMeyars['E'], 
           c=resultMeyars['PredictedCluster'], s=70,cmap='hsv')
ax.legend()
plt.show()



sns.pairplot(resultMeyars, kind="scatter",hue='PredictedCluster')
plt.show()



from matplotlib import pyplot as plt
import seaborn as sns    
from matplotlib import pyplot


fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(resultMeyars['E'], resultMeyars['N'], resultMeyars['K'],
           c=resultMeyars['PredictedCluster'], s=70,cmap='viridis')
pyplot.legend()
pyplot.show()





plt.subplot(321)
plt.scatter(resultMeyars['E'], resultMeyars['N'], s=80, c=resultMeyars['PredictedCluster'], marker=">")

plt.subplot(322)
plt.scatter(resultMeyars['E'], resultMeyars['K'], s=80, c=resultMeyars['PredictedCluster'], marker=(5, 0))



plt.show()


##############