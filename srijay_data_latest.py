# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 01:11:58 2023

@author: fayya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from CoupledScatterPlot import *
df0 = pd.read_csv('/home/u1876024/Downloads/cellular_proteinexpressions_raw.csv')#('./srijaydata.xlsx')
df0 = df0.dropna()
house_cols = ['spot_name','image_id','VisSpot', 'Location_Center_X', 'Location_Center_Y']
expression_cols = ['OLIG2', 'MET', 'cMYC', 'PDGFRa', 'CD14', 'CD24', 'GFAP', 'NESTIN', 'EGFR', 'CDK4', 'SMAa', 'HLADR', 'S100B', 'CD206', 'CD11b', 'YKL40', 'CD68', 'TCIRG1', 'TMEM119', 'pERK', 'IBA1', 'CD31', 'DNA3', 'SOX2', 'PTEN', 'MHCI', 'MCT4', 'HIF1a', 'CD74', 'CD44', 'KI67', 'CD16', 'P2RY12', 'SOX10', 'CD163', 'CD11c', 'DNA1', 'VISTA']
df = df0[house_cols+expression_cols]

def max_distance(group):    
    coordinates = group[['Location_Center_X', 'Location_Center_Y']].values    
    distances = pdist(coordinates)    
    if len(distances)==0: return 0
    D = np.max(distances)
    return D

Dmax = df.groupby('VisSpot').apply(max_distance)
valid_spots = list(Dmax[(Dmax >= 40) & (Dmax <= 60)].index)
df = df[df['VisSpot'].isin(valid_spots)]



#%%
def CatTransform(z):
    zlog = np.log10(z[z>0])
    p25,p75 = zlog.quantile([0.25,0.75])
    def map_to_category(value):
        if value < p25:
            return 1
        elif value <= p75:
            return 2
        else:
            return 3
    tzlog = zlog#.apply(map_to_category)
    tz = z.copy()
    tz[tz>0] = tzlog
    return tz
def ZSSTransform(z):
    zlog = np.log10(z[z>0])
    tzlog = (zlog-np.mean(zlog))/np.std(zlog)
    tzlog_min = np.min(tzlog)-3 #zero is assumed to be at least 1000 times smaller than the lowest non-zero value
    tz = z.copy()
    tz[tz>0] = tzlog
    tz[tz==0] = tzlog_min
    return tz
    
def normalize_median_range(data):
    median = np.median(data)
    Ql = np.percentile(data, 10)
    Qr = np.percentile(data, 90)
    IQR = Qr - Ql + 1e-10
    normalized_data = (data - median) / IQR
    return normalized_data

def ArcsinhTransform(z):    
    zs = normalize_median_range(z)
    zt = np.arcsinh(3*zs)    
    return zt

def Standardize(z):        
    return (z-np.mean(z))/np.std(z)

def cluster_group(group_df):
    group_df['cluster'] = KMeans(n_clusters=12).fit_predict(group_df[expression_cols])
    return group_df
#%% Apply the transform
plotHist(df[expression_cols])

Xt = df[expression_cols+['image_id']]
for col in expression_cols:
    Xt[col] = Xt.groupby('image_id')[col].transform(ArcsinhTransform)

Xt = Xt[expression_cols]
Xt = df[expression_cols].apply(ZSSTransform)
# Xt = Xt.transform(Standardize)
plotHist(Xt)

C = df.image_id
Xt[house_cols]=df[house_cols]

Xt = Xt.groupby('image_id').apply(cluster_group) #apply separate clustering on each image -- so cluster x for one image is not the same as cluster x for another one
#plot2D(Xt[expression_cols][C=='A1'],Xt.cluster[C=='A1'])

#sns.scatterplot(data=df[df.image_id == 'A1'], x='Location_Center_X', y='Location_Center_Y', hue='NESTIN', palette='viridis')

#%% Batch correction

from combat.pycombat import pycombat
X_corrected = pycombat(Xt[expression_cols].T,C).T
X_corrected[Xt.keys().difference(expression_cols)]=Xt[Xt.keys().difference(expression_cols)]#restore all remaining columns

#plot2D(X[expression_cols],C,'Original');
#plot2D(Xt[expression_cols],C,'Log Transformed')
#plot2D(X_corrected[expression_cols],C,'Log transformed and Corrected')

Xt.to_csv('fayyaz_transformed_zss.csv',index = False)
X_corrected.to_csv('fayyaz_corrected_zss.csv',index = False)
1/0
#%% Visualization
import umap
X2_original = umap.UMAP().fit_transform(df[expression_cols])
X2_transformed = umap.UMAP().fit_transform(Xt[expression_cols])
X2_corrected = umap.UMAP().fit_transform(X_corrected[expression_cols])

X_corrected_patched = X_corrected.groupby('VisSpot')[expression_cols].mean()
X2_corrected_patched = umap.UMAP().fit_transform(X_corrected_patched)
#%%
from CoupledScatterPlot import CSCPlot
CSCPlot(X2_original,X2_transformed,clusters = C)
CSCPlot(X2_original,X2_corrected,clusters = C)
plot2D(np.array(X2_corrected_patched),[x[-2:] for x in X_corrected_patched.index])