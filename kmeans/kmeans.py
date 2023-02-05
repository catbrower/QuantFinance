from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pypfopt import risk_models

def custom_covariance(X, threshold=1e-5):
    # SVD
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    U, s, V = np.linalg.svd(X, full_matrices=False)

    # Remove small singular values
    s = s[s > threshold]

    # recombine to remove linear dependencies
    X_reduced = np.dot(U, np.dot(np.diag(s), V))

    # calculate covariance
    cov_matrix = risk_models.exp_cov(X_reduced, returns_data=True, span=252)

    return cov_matrix

def custom_pca(X):
    cov_matrix = custom_covariance(X)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # sort eigenvectors and eigen values in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # choose top eigenvetors
    k = 2
    eigenvectors = eigenvectors[:, :k]

    # transform the data into the first k principle components
    X_transformed = X @ eigenvectors

def write_file(fileName, data, header):
    with open(fileName, 'w') as file:
        file.write(header)
        for _row in data:
            row = [str(x) for x in _row]
            file.write(','.join(row) + '\n')

# So I think what we need to do is calculate the returns for all the stocks
# and put them into a 2d matrix

data = pd.read_csv('/home/catherine/stock_data/test_data.csv')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
# data = data.set_index('date')
assets = data['ticker'].unique()[:252]

# data = data.set_index(['region', 'ticker', 'date'])
# for asset in assets:
#     close = data.loc[data.index.get_level_values(level=1) == asset]['close']
#     close = np.log(close)
#     data.loc[data.index.get_level_values(level=1) == asset, 'returns'] = close.pct_change().values
# data.to_csv('/home/catherine/stock_data/test_data.csv')
# quit()

use_zscore = True
# returns_data = pd.DataFrame({'date': data.index.unique()})
# returns_data.set_index('date')
returns_series = []
for asset in assets:
    # drop the first entry b/c inf
    returns = pd.Series(np.log(data.loc[data['ticker'] == asset, 'close']).values, index=data.loc[data['ticker'] == asset, 'date'].values)
    returns = returns.pct_change()
    returns = returns.iloc[1:]

    # if len([x for x in returns if x == float('inf') or x == float('-inf') or x == float('nan')]) > 0:
    #     continue
    # else:
    if use_zscore:
        returns = (returns - np.mean(returns)) / np.std(returns)
        # returns_data[asset] = pd.Series(returns)
        returns_series.append(returns)
    else:
        returns_series.append(returns)
        # returns_data[asset] = returns

returns_data = pd.concat(returns_series, axis=1, join='outer')
returns_data = returns_data.dropna(axis=1)
# normalizer = Normalizer()   
# returns_data = normalizer.fit_transform(np.array(returns_data)).T
# returns_data = np.array(returns_data).T
# returns_data = pd.DataFrame(returns_data)

# PCA
# pca = PCA(n_components=2)
# pca.fit(returns_data)
# returns_pca = pca.transform(returns_data)
# returns_pca = custom_pca(returns_data)

# potential problem here as all points are weighted equally
# TODO look into coviariance matricies that are time sensitive
# Potential solution: apply an exponential weighting function to each oberservation
# in the same way you do with EMA, see:
# https://reasonabledeviations.com/2018/08/15/exponential-covariance/
# this soluton looks to be in pyportfolioopt already
# cov_matrix = LedoitWolf().fit(returns_data).covariance_
# 

# do the labels / clusters thing i dunno
# cluster_dict = {}
# for i, label in enumerate(stock_labels):
#     cluster_dict[stock]

### Elbow Method
# Determine optimal num of clusters
# wcss = []

# # fit kmeans and calculate wcss for each cluster
# cov_matrix_2 = risk_models.sample_cov(returns_data, returns_data=True)
# xs = np.arange(50)
# xs = xs + 1
# for i in xs:
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=3000, n_init=10)
#     kmeans.fit(cov_matrix_2)
#     wcss.append(kmeans.inertia_)

# # PLot it
# plt.plot(np.log(xs), np.log(wcss))
# plt.show(block=True)

# print()

labels_data = []
def get_labels(returns):
    cov_matrix = risk_models.exp_cov(returns_data, returns_data=True, span=252)
    kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=3000, n_init=10)
    kmeans.fit(cov_matrix.values)
    return kmeans.labels_


# From the elbow method it looks like 7 is a good number
# Let's see how it looks when applying it through time
date_now = min(data['date']) + timedelta(days=365)
date_end = max(data['date'])
while date_now <= date_end:
    try:
        selection = returns_data[(returns_data.index >= date_now - timedelta(days=365)) & (returns_data.index < date_now)]
        labels = get_labels(selection.values)
        labels_data.append([max(selection.index)] + list(labels))
    except Exception as err:
        print('exception encountered')
    
    date_now += timedelta(days=1)
    # labels = returns_data.rolling(252).apply(lambda values : get_labels(values))

write_file('labels_data.csv', labels_data, ','.join(['date'] + list(assets)) + '\n')
# np.savetxt('labels_data.csv', labels_data, delimiter=',', header=['date'] + ','.join(assets))
print()