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
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)



print(x)


from sklearn.cluster import Birch

brc = Birch(branching_factor=10, n_clusters=8, threshold=0.5,compute_labels=True)
brc_predict=brc.fit_predict(x)


print(brc_predict)

from sklearn import metrics
#print(metrics.silhouette_score(x, brc_predict, metric='euclidean'))
print("Silhouette Score: %0.3f"% metrics.silhouette_score(x, brc_predict, metric='euclidean'))
print("Calinski-Harabasz Index: %0.3f"% metrics.calinski_harabaz_score(x, brc_predict))
#print("Calinski-Harabasz Index: %0.3f" %davies_bouldin_score(x, brc_predict))
#print(contingency_matrix(x, brc_predict))




# Add cluster column to initial DF
predictdf=pd.DataFrame(brc_predict)
predictdf.columns=['PredictedCluster']

result = pd.concat([df, predictdf], axis=1)
#print(result)

writer=pd.ExcelWriter("resultBirch.xlsx",engine='xlsxwriter')
result.to_excel(writer, sheet_name='new')
writer.close()
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


print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))

print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))"""

