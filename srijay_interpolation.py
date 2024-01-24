# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:41:23 2024

@author: fayya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
df = pd.read_csv('C:/Users/fayya/Downloads/interpolation_data.csv')
K = df.keys()
df = df[K[np.argsort([-np.abs(df[k][9]-df[k][0]) for k in K])]].T
df.to_excel('interpolation_results.xlsx')
labels = df.index
P = np.array(df)

# Load images
images = [Image.open(f"{i}.png") for i in range(1, 11)]

# Set up the plot
fig_width = 20
fig_height = 2 * len(P)  # Increased height to make heatmap cells bigger
fig, axarr = plt.subplots(2, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05, 'wspace': 0.05})

# Max and min values for normalization in heatmap
min_val, max_val = np.min(P), np.max(P)

for i in range(10):
    # Show images
    axarr[0, i].imshow(images[i])
    axarr[0, i].axis('off')

    # Show heatmaps
    heatmap = axarr[1, i].imshow(P[:, i].reshape(-1, 1), aspect='auto',cmap='bwr', vmin=min_val, vmax=max_val)
    axarr[1, i].axis('off')

# Add row labels on the left
for idx, label in enumerate(labels):
    axarr[1, 0].text(-0.5, idx, label, ha='right', va='center', transform=axarr[1, 0].transData)

# Add a colorbar
fig.subplots_adjust(right=0.85, left=0.15)
cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
fig.colorbar(heatmap, cax=cbar_ax)

# Save the figure
plt.savefig("output.png", bbox_inches='tight')
plt.show()

