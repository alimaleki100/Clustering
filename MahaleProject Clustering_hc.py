# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:40:01 2019

@author: session1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy.cluster.hierarchy as shc



df=pd.read_excel('Dataset3-2.xls')

############################################Cleaning###########################################################

# Taking care of missing data
df=df.fillna(value=0)

############################################DefineTargetData###########################################################

#Set cluster data
x=df.iloc[:,5:].values
xforest=df.iloc[:,5:].values


#to use in feature importance
xFeatures=df.iloc[:,5:]
xColumns=pd.DataFrame(xFeatures.columns)


####################DENDOGRAM

plt.figure(figsize=(10, 10))  
plt.title("Dendograms")  
dend = shc.dendrogram(shc.linkage(x, method='ward',)) 



x=pd.DataFrame(x)



seaborn.distplot(x[0], bins=20,color='blue')




############################################encoding###########################################################

# Encoding the Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [9])
x = onehotencoder.fit_transform(x).toarray()

onehotencoder = OneHotEncoder(categorical_features = [17])
x = onehotencoder.fit_transform(x).toarray()


############################################FeatureScaling###########################################################from sklearn.preprocessing import StandardScaler

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

"""
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)




############################################ClusterModeling###########################################################

from sklearn.cluster import AgglomerativeClustering
#linkage : {“ward”, “complete”, “average”, “single”}, optional (default=”ward”)
ac = AgglomerativeClustering(n_clusters=7, linkage='ward')
ac_predict = ac.fit_predict(x)
print(ac_predict)


seaborn.distplot(ac_predict,kde=False)


############################################Metrics###########################################################

from sklearn import metrics

from sklearn.metrics import pairwise_distances
print("Silhouette Score: %0.3f"% metrics.silhouette_score(x, ac_predict, metric='euclidean'))
print("Calinski-Harabaz Index: %0.3f"% metrics.calinski_harabaz_score(x, ac_predict))
 





############################################SavingInXlsx###########################################################

# Add cluster column to initial DF
predictdf=pd.DataFrame(ac_predict)
predictdf.columns=['PredictedCluster']

result = pd.concat([df, predictdf], axis=1)
writer=pd.ExcelWriter("resultHC.xlsx",engine='xlsxwriter')
result.to_excel(writer, sheet_name='new')
writer.close()



############################################Mantaghe###########################################################
x=pd.DataFrame(x)

xMantaghe = pd.concat([x, df['Mantaghe']], axis=1, sort=False)
heatMapMantaghe=xMantaghe
xMantaghe = pd.concat([xMantaghe, predictdf], axis=1, sort=False)
xMantaghe = pd.concat([xMantaghe, df['Mahale']], axis=1, sort=False)
print(xMantaghe)


mantaghe=5

resultMantaghe=xMantaghe[xMantaghe['Mantaghe']==mantaghe]
print(resultMantaghe)
resultMantaghe=resultMantaghe.set_index('Mahale')




dfMantaghe=pd.concat([df,predictdf],axis=1,sort=False)
mantaghedf=5

resultdfMantaghe=dfMantaghe[dfMantaghe['Mantaghe']==mantaghedf]
resultdfMantaghe=resultdfMantaghe.set_index('Mahale')

####Dendogram of Manataghe#####
from scipy.cluster import hierarchy

Z = hierarchy.linkage(resultMantaghe, 'ward')
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=resultMantaghe.index)





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

forest.fit(xforest,ac_predict)
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










############################################Visualization###########################################################

plt.figure(figsize=(5, 5))  
plt.scatter(x[:,37], x[:,23], c=ac.labels_, cmap='rainbow')
ax2 = result.plot.scatter(x='K1', y='E2', c='PredictedCluster',colormap='viridis',figsize=(10,10))


Z = hierarchy.linkage(x, 'ward')
x=pd.DataFrame(x)
# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)



import seaborn as sns; sns.set(color_codes=True)
g = sns.clustermap(x, metric="correlation",method="single",cmap="mako",figsize=(10, 10))

def plot_clustered_dataset(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    markers = ['o', 'd', '^', 'x', '1', '2', '3', 's']
    colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y', '#cccfff']

    for i in range(nb_samples):
        ax.scatter(X[i, 0], X[i, 1], marker=markers[Y[i]], color=colors[Y[i]])

    plt.show()
    # Show the clustered dataset
plot_clustered_dataset(x, ac_predict)












from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['K11'], result['K10'], result['K2'], 
           c=result['PredictedCluster'], s=70,cmap='viridis')
ax.view_init(30, 30)
ax.legend()
ax.set_xlabel('Garmayesh')
ax.set_ylabel('Sarmayesh')
ax.set_zlabel('Tabaghat')

plt.show()







from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['N4'], result['N7'], result['N8'], 
           c=result['PredictedCluster'], s=70,cmap='viridis')
ax.view_init(30, 30)
ax.legend()
ax.set_xlabel('Ketab')
ax.set_ylabel('SalonVarzeshi')
ax.set_zlabel('ZaminChaman')

plt.show()


import seaborn as sns
sns.pairplot(result, kind="scatter",palette="husl",hue='PredictedCluster',vars=['N4','N7','N8','PredictedCluster'],height=5)



sns.pairplot(result, kind="scatter",palette="husl",hue='PredictedCluster',vars=['K1','E2','N2','N3','PredictedCluster'],height=5)


from scipy.cluster import hierarchy






import seaborn as sns; sns.set(color_codes=True)
g = sns.clustermap(x, metric="correlation",method="single",cmap="hsv",figsize=(20, 10))

resultMeyarArray = np.asarray(resultMeyars)






# Without regression fit:
#sns.regplot(x=resultMeyars['E'], y=resultMeyars['N'], fit_reg=False,color=resultMeyars['PredictedCluster'])

# without regression
sns.pairplot(result, kind="scatter",palette="husl",hue='PredictedCluster')
sns.pairplot(result, kind="reg",palette="husl",vars=['K','E','A','N'])

plt.show()


sns.clustermap(x.corr(), center=0, cmap="vlag",linewidths=.75, figsize=(13, 13))


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(resultMeyars['E'], resultMeyars['K'], resultMeyars['A'], 
           c=resultMeyars['PredictedCluster'], s=70,cmap='viridis')
ax.view_init(50, 150)
ax.legend()
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


"""



