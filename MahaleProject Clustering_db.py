# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:40:01 2019

@author: session1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel('Dataset3-2.xls')


# Taking care of missing data
df=df.fillna(value=0)

# review dataframe
print(df)
print(df.dtypes)





#Set cluster data
x=df.iloc[:,5:].values
x=pd.DataFrame(x)
#print(x)


# Encoding the Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [9])
x = onehotencoder.fit_transform(x).toarray()

onehotencoder = OneHotEncoder(categorical_features = [17])
x = onehotencoder.fit_transform(x).toarray()

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)
"""
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)


from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=2, min_samples=2)
#dbs = DBSCAN(eps=0.5, min_samples=540)
db_predict= dbs.fit_predict(x)

print(db_predict)
from sklearn import metrics

from sklearn.metrics import pairwise_distances
print("Silhouette Score: %0.3f"% metrics.silhouette_score(x, db_predict, metric='euclidean'))
print("Calinski-Harabaz Index: %0.3f"% metrics.calinski_harabaz_score(x, db_predict))
 


"""



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
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

#Generate the Cluster column
y_kmeans = kmeans.fit_predict(x)

print(y_kmeans)

# Add cluster column to initial DF
predictdf=pd.DataFrame(y_kmeans)
predictdf.columns=['PredictedCluster']

result = pd.concat([df, predictdf], axis=1, sort=False)
#print(result)

# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 23], x[y_kmeans == 0, 25], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 23], x[y_kmeans == 1, 25], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 23], x[y_kmeans == 2, 25], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 23], x[y_kmeans == 3, 25], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()




writer=pd.ExcelWriter("result2.xlsx",engine='xlsxwriter')
result.to_excel(writer, sheet_name='new')
writer.close()


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


print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))

print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))"""

