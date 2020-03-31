from sklearn import cluster
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def k_mean(X_data, num_clus):
	k_means = cluster.KMeans(n_clusters=num_clus)
	k_means.fit(X_data)
	print("total label :" , str(len(k_means.labels_)))
	# print(k_means.labels_)
	return k_means

def pca(X_data, n_comp):
	pca = PCA(n_components=n_comp)
	pca.fit(X_data)
	# print(pca.components_[:10])
	print("variance : ", pca.explained_variance_)
	print("pca shape : ", pca.components_.shape)
	return np.transpose(pca.components_)