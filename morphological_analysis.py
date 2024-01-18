# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:51:27 2024

@author: fayya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for the datasets
generated_file = r'C:/Users/fayya/OneDrive - University of Warwick/Desktop/coding/morph_features_generated_C.csv'
true_file = 'C:/Users/fayya/OneDrive - University of Warwick/Desktop/coding/morph_features_all 1.csv'

# Columns in the dataset that are not features
non_feats = ['Unnamed: 0', 'spot_name', 'image_id', 'VisSpot']

# Loading the generated and true data into Pandas DataFrames
G = pd.read_csv(generated_file)
T = pd.read_csv(true_file)

# Extracting the feature columns
feats = G.keys().difference(non_feats)

# Grouping data by 'spot_name' and calculating the mean of features
Gf = G.groupby('spot_name')[feats].mean()
Tf = T.groupby('spot_name')[feats].mean()

#%%
from sklearn.preprocessing import StandardScaler

# Stacking the feature means vertically to form a single dataset for scaling
X = np.vstack((Gf, Tf))

# Applying Standard Scaler to normalize the data
ss = StandardScaler().fit(X)

# Transforming both datasets using the fitted scaler
Gft, Tft = ss.transform(Gf), ss.transform(Tf)

# Extracting the number of features
n_feats = len(feats)

# Converting the transformed data back to Pandas DataFrames
Gft = pd.DataFrame(Gft[:, :n_feats], index=Gf.index)
Tft = pd.DataFrame(Tft[:, :n_feats], index=Tf.index)

# Merging the two datasets on their index
df = pd.merge(Tft, Gft, left_index=True, right_index=True)
X = np.array(df)

# Splitting the merged data back into 'true' and 'generated' datasets
Xt, Xg = X[:, :n_feats], X[:, n_feats:]

# Calculating the Euclidean distance between corresponding rows in Xt and Xg
D = np.linalg.norm(Xt - Xg, axis=1)

#%%
def average_distance(matrix, M):
    """
    Function to calculate the average distance of each row in a matrix to M randomly selected other rows.
    :param matrix: NumPy array representing the data matrix.
    :param M: Number of random rows to select for distance calculation.
    :return: Array of average distances.
    """
    N, d = matrix.shape
    avg_distances = np.zeros(N)

    for i in range(N):
        # Select M random rows, excluding the current row i
        random_indices = np.random.choice(np.delete(np.arange(N), i), M, replace=False)
        selected_rows = matrix[random_indices]

        # Calculate distances from row i to the selected M rows
        distances = np.linalg.norm(matrix[i] - selected_rows, axis=1)

        # Compute the average distance
        avg_distances[i] = np.mean(distances)

    return avg_distances

# Calculating the average distances for the 'true' dataset
Dbar = average_distance(Xt, M=100)

#%%
# Creating a boxplot to compare distributions of D and Dbar
plt.boxplot([D, Dbar])

from scipy import stats

# Performing a Wilcoxon signed-rank test on D and Dbar.
# Hypothesis: For each image patch, the distance between morphological features 
# of cells in the original image and its generated counterpart (D) is less than 
# the average distance between morphological features of cells in that image 
# and randomly selected original patches from the same whole slide image (Dbar).
# This test aims to statistically verify if the generated image patches 
# maintain closer morphological feature similarity to their original counterparts 
# compared to random original patches, indicating accuracy in feature preservation during generation.
w_stat, p_value = stats.wilcoxon(D, Dbar, alternative='less')

# Printing the test statistics
print("W-statistic:", w_stat)
print("P-value:", p_value)
