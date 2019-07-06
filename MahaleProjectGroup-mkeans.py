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
print(x)


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
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    print(wcss)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Train the model
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
#Generate the Cluster column
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

############################################Metrics###########################################################

print("Silhouette Score: %0.3f"% metrics.silhouette_score(x, y_kmeans, metric='euclidean'))
print("Calinski-Harabaz Index: %0.3f"% metrics.calinski_harabaz_score(x, y_kmeans))

#print(result)
# K  0 Kalbadi
# E  1 Emkanat
# N 2 NirooyeEnsani
# A 3 Amalkardi
"""
#y_kmeans==0 ---> No of Cluster, ,0--> No of Meyar to visualize
# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1],s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Amalkard')
plt.ylabel('Niroo Ensani')
plt.legend()
plt.show()
"""

############################################SavingInXlsx###########################################################

x=pd.DataFrame(x)
# Add cluster column to initial DF
predictdf=pd.DataFrame(y_kmeans)
predictdf.columns=['PredictedCluster']

# Save Predict Cluster with Features

resultFeatures = pd.concat([df, predictdf], axis=1, sort=False)
writer=pd.ExcelWriter("resultKMeansFG.xlsx",engine='xlsxwriter')
resultFeatures.to_excel(writer, sheet_name='new')
writer.close()

# Save Predict Cluster with Meyars
resultMeyars = pd.concat([x, predictdf], axis=1, sort=False)
writer=pd.ExcelWriter("resultKMeansGG.xlsx",engine='xlsxwriter')
resultMeyars.to_excel(writer, sheet_name='new')
writer.close()

ax2 = resultMeyars.plot.scatter(x='K', y='E', c='PredictedCluster',colormap='viridis',figsize=(10,10))


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
ax = sns.scatterplot(x=0, y=1, hue="PredictedCluster",data=resultMeyars)

############################################Vis###########################################################


"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


lrx = result.iloc[:,5:-1].values
lry = result.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(
    lrx,lry, test_size = 0.25, random_state = 0)

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

coefs=classifier.coef_[0]
print(coefs)
top_three = np.argpartition(coefs, -34)[-34:]
print(top_three)
top_three_sorted=top_three[np.argsort(coefs[top_three])]
print(result.columns[top_three])

"""
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

 

 
# Calculate the distance between each sample
Z = hierarchy.linkage(x, 'ward')
x=pd.DataFrame(x)
# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)


#HeatMap
import seaborn as sns; sns.set(color_codes=True)
g = sns.clustermap(x, metric="correlation",method="single",cmap="mako",figsize=(10, 10))

############################################Testing###########################################################


xMantaghe = pd.concat([x, df['Mantaghe']], axis=1, sort=False)
heatMapMantaghe=xMantaghe
xMantaghe = pd.concat([xMantaghe, predictdf], axis=1, sort=False)
xMantaghe = pd.concat([xMantaghe, df['Mahale']], axis=1, sort=False)
print(xMantaghe)


mantaghe=1

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
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x=2, y=3, hue='PredictedCluster',data=resultMantaghe)



# Without regression fit:
sns.regplot(x=resultMantaghe[0], y=resultMantaghe[1], fit_reg=False)
plt.show()

# without regression
sns.pairplot(resultMantaghe, kind="scatter")
plt.show()


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(resultMantaghe[0], resultMantaghe[1], resultMantaghe[2], c='red', s=60)
ax.view_init(30, 185)
plt.show()




sns.pairplot(resultMantaghe, kind="scatter",hue='PredictedCluster')
plt.show()



from matplotlib import pyplot as plt
import seaborn as sns    

plt.figure(figsize=(15,16))
sns.countplot(data=resultMantaghe,x=resultMantaghe[0])




from matplotlib import pyplot


fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(resultMantaghe['E'], resultMantaghe['N'], resultMantaghe['K'])
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