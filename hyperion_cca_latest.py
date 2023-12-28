#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:49:51 2023

@author: u1876024
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CoupledScatterPlot import CSCPlot
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from umap import UMAP
from combat.pycombat import pycombat

#%% Load data
bdir = '/home/u1876024/Downloads/'
#dfX = pd.read_csv(bdir+'cellular_proteinexpressions_raw.csv')#cellular_proteinexpressions_batchcorrected.csv
#dfX = pd.read_csv(bdir+'cellular_proteinexpressions_batchcorrected.csv')#
dfX = pd.read_csv('fayyaz_corrected_zss.csv')
#dfX = pd.read_csv('fayyaz_transformed_zss.csv')
dfX = dfX.dropna()
#non_expression = ['Unnamed: 0', 'VisSpot', 'Location_Center_X', 'Location_Center_Y','Unnamed: 43','Unnamed: 42','Unnamed: 0.1', 'spot_name', 'image_id']
#expression_cols = list(set(dfX.keys()).difference(non_expression))
expression_cols = ['OLIG2', 'MET', 'cMYC', 'PDGFRa', 'CD14', 'CD24', 'GFAP', 'NESTIN', 'EGFR', 'CDK4', 'SMAa', 'HLADR', 'S100B', 'CD206', 'CD11b', 'YKL40', 'CD68', 'TCIRG1', 'TMEM119', 'pERK', 'IBA1', 'CD31', 'DNA3', 'SOX2', 'PTEN', 'MHCI', 'MCT4', 'HIF1a', 'CD74', 'CD44', 'KI67', 'CD16', 'P2RY12', 'SOX10', 'CD163', 'CD11c', 'DNA1', 'VISTA']
dfY = pd.read_csv(bdir+'morph_features_all.csv')
non_morpho = ['spot_name', 'image_id', 'VisSpot','Unnamed: 0']
morpho_cols = list(set(dfY.keys()).difference(non_morpho))
dfY = dfY.dropna()
C = np.array([x[-2:] for x in list(dfY.VisSpot)]) #image based clusters
Y0 = dfY[morpho_cols]
#dfY[morpho_cols] = pycombat(((Y0-Y0.mean())/Y0.std()).T,C).T
X = dfX.groupby('VisSpot')[expression_cols].mean()
Y = dfY.groupby('VisSpot')[morpho_cols].mean()
dfXY = pd.merge(X, Y, on='VisSpot', how='inner')
X = dfXY[expression_cols]
Y = dfXY[morpho_cols]
C_patch = [x[-2:] for x in X.index]
#X = pycombat(X[expression_cols].T,C_patch).T
#Y = pycombat(Y[morpho_cols].T,C_patch).T

#%% Standardize and PCA Preprocessing

pipelineX = Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=0.97,whiten=True))]).fit(X)
X_r = pipelineX.transform(X)

pipelineY = Pipeline(steps=[('scaler',StandardScaler()),('pca',PCA(n_components=0.97,whiten=True))]).fit(Y)
Y_r = pipelineY.transform(Y)

#%% Canonical Correlation Analysis
cca = CCA(n_components=12)
cca.fit(X_r, Y_r)
X_c, Y_c = cca.transform(X_r, Y_r)
#X_c,Y_c = X_c[:,1:],Y_c[:,1:]
#X_c,Y_c = X_r,Y_r
#X_c = StandardScaler().fit_transform(X_c)
#Y_c = StandardScaler().fit_transform(Y_c)
# Plot correlations between corresponding features after alignment
plt.plot([np.corrcoef(X_c[:,d],Y_c[:,d])[0,1] for d in range(min(Y_c.shape[1],X_c.shape[1]))])

#%% Reduction to 2D for visualization only
reducer =UMAP # TSNE #   PCA #
X2 = reducer().fit_transform(X_c)
Y2 = reducer().fit_transform(Y_c)
#%% Clustering and visualization
#X2,Y2 = X_c,Y_c
# Number of clusters
n_clusters = 15  # Adjust based on your requirements
#clusters = kmeans.fit_predict(np.hstack((X_c, Y_c)))
clusters = GaussianMixture(n_clusters).fit_predict(Y_c)#np.hstack((X_c, Y_c)))
clusters = C_patch

CSCPlot(X2, Y2, np.array(clusters))

#%% train predictors to predict Y from X using LOGO

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np

data = pd.merge(X, Y, left_index=True, right_index=True)
data['image_id']=[x[-2:] for x in dfXY.index]

# Splitting features and targets
X_features,Y_targets = data[morpho_cols],data[expression_cols]
#X_features,Y_targets = pd.DataFrame(Y_r),data[expression_cols]


# Leave-One-Image-Out Cross-Validation
logo = LeaveOneGroupOut()
groups = data.image_id

#groups = np.random.randint(0,4,size=len(data.image_id))

# Choose a suitable model
#model = MultiOutputRegressor(RandomForestRegressor(n_estimators = 10))
model = MultiOutputRegressor(LinearRegression())

# To store the average correlations
average_correlations = []
CC = []
# Iterate over each group (image_id)
for train_index, test_index in logo.split(X_features, Y_targets, groups):
    X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
    Y_train, Y_test = Y_targets.iloc[train_index], Y_targets.iloc[test_index]

    # Train the model
    model.fit(X_train, Y_train)

    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate correlations for each field and take the average
    correlations = [np.corrcoef(Y_test.iloc[:, i], predictions[:, i])[0, 1] for i in range(Y_test.shape[1])]
    CC.append(correlations)
    avg_corr = np.nanmean(correlations)  # Using nanmean to handle any NaNs in correlations
    average_correlations.append(avg_corr)
    #print(f"Average Correlation for image_id {groups.iloc[test_index[0]]}: {avg_corr}")
    
df_corr = pd.DataFrame(CC,columns = list(Y_targets.keys()))
# Overall average correlation
overall_avg_corr = np.mean(average_correlations)
print(f"Overall Average Correlation: {overall_avg_corr}")
print(df_corr.mean(axis=0).sort_values(ascending=False).head(10))
# Note: Correlation might not be applicable in some cases (e.g., non-linear relationships or categorical data).
