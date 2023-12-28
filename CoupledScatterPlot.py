#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:33:45 2023

@author: u1876024
"""
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CustomJS, BoxSelectTool, TapTool,LassoSelectTool,PolySelectTool
from bokeh.layouts import row
from bokeh.transform import factor_cmap
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns    

def plot2D(X,C,title = '',xidx=0,yidx=1):
    if X.shape[1]>2:
        import umap
        reducer = umap.UMAP()
        #reducer = PCA()
        reducer = reducer.fit(X)    
        Z = reducer.transform(X)
    else:
        Z = X
    dfZ = pd.DataFrame(Z[:,[xidx,yidx]], columns=['X', 'Y']) #select PCA axes here
    dfZ['Group'] = np.array(C)    
    sns.pairplot(data=dfZ, hue='Group')
    #sns.pairplot(data=dfZ, kind = 'kde',hue='Group')
    #sns.kdeplot(data=dfZ, x='X', y='Y', hue='Group', fill=True);
    plt.title(title)
    return Z

def plotHist(X,title=''):

    fig, axes = plt.subplots(nrows=3, ncols=13, figsize=(20, 8))
    K = list(X.keys())
    for i, ax in enumerate(axes.flatten()):
        if i < len(K):
            c = K[i]
            ax.hist(X[c],density = True);#ax.title(c)
            ax.set_title(c)
            #ax.axis('off')
            ax.yaxis.set_visible(False)
        else:
            # Hide unused subplots
            ax.axis('off')
    plt.title(title)
    plt.show()
    
def generate_distinct_colors(n):
    """
    Generate n visually distinct colors in RGB format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7  # Saturation and lightness can be adjusted as needed
        lightness = 0.5
        rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert to hexadecimal for Bokeh
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255))
        colors.append(hex_color)
    return colors
def CSCPlot(X_c,Y_c = None,clusters = None,ofname = "linked_scatter_with_region_select.html"): 
    if Y_c is None:
        Y_c = X_c


    if clusters is None:
        clusters = np.ones(X_c.shape[0])
    size = 3
    x,y = 0,1 #dimesnions to show

    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(x1=X_c[:, x], y1=X_c[:, y], x2=Y_c[:, x], y2=Y_c[:, y],cluster=clusters.astype(str)))
    n_clusters =  len(np.unique(clusters))

    #colors = ["navy", "firebrick", "olive", "goldenrod", "purple"]  # Add more colors if needed
    colors = generate_distinct_colors(n_clusters)
    # Create two scatter plots
    p1 = figure(plot_width=400, plot_height=400, tools=[BoxSelectTool(), TapTool(),PolySelectTool(), 'pan','wheel_zoom', 'zoom_in', 'zoom_out', 'reset'], title="Components of X")
    p1.circle('x1', 'y1', source=source, size=size, color=factor_cmap('cluster', palette=colors, factors=np.unique(clusters).astype(str).tolist()), alpha=0.3, selection_color="black", nonselection_alpha=0.1)
    
    p2 = figure(plot_width=400, plot_height=400, tools=[BoxSelectTool(), TapTool(),PolySelectTool(), 'pan','wheel_zoom','zoom_in', 'zoom_out', 'reset'], title="Components of Y")
    p2.circle('x2', 'y2', source=source, size=size, color=factor_cmap('cluster', palette=colors, factors=np.unique(clusters).astype(str).tolist()), alpha=0.3, selection_color="black", nonselection_alpha=0.1)
    
    # JavaScript to link selections
    code = '''
    if (cb_obj.indices.length > 0) {
        source.selected.indices = cb_obj.indices;
    } else {
        source.selected.indices = [];
    }
    '''
    zoom_code = '''
    if (cb_obj.start != source.start || cb_obj.end != source.end) {
        source.start = cb_obj.start;
        source.end = cb_obj.end;
    }
    '''
    zoom_callback_x = CustomJS(args=dict(source=p2.x_range), code=zoom_code)
    zoom_callback_y = CustomJS(args=dict(source=p2.y_range), code=zoom_code)
    # Attach callbacks to x and y ranges
    p1.x_range.js_on_change('start', zoom_callback_x)
    p1.x_range.js_on_change('end', zoom_callback_x)
    p1.y_range.js_on_change('start', zoom_callback_y)
    p1.y_range.js_on_change('end', zoom_callback_y)

    # Attach the callback to both plots
    callback = CustomJS(args=dict(source=source), code=code)
    source.selected.js_on_change('indices', callback)
    
    # Arrange plots in a row
    layout = row(p1, p2)
    
    # Output to static HTML file
    output_file(ofname)
    
    # Show results
    show(layout)