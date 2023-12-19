# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 02:21:08 2023

@author: fayya
"""
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CustomJS, BoxSelectTool, TapTool
from bokeh.layouts import row
from sklearn.cross_decomposition import CCA
from sklearn.datasets import load_iris

# Load Iris dataset
data = load_iris()
X = data.data[:, :2]  # First two features
Y = data.data[:, 2:]  # Last two features

# Initialize and fit CCA
cca = CCA(n_components=2)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(x1=X_c[:, 0], y1=X_c[:, 1], x2=Y_c[:, 0], y2=Y_c[:, 1]))

# Create two scatter plots
p1 = figure(plot_width=400, plot_height=400, tools=[BoxSelectTool(), TapTool()], title="Components of X")
p1.circle('x1', 'y1', source=source, size=10, color="navy", alpha=0.5, selection_color="orange", nonselection_alpha=0.1)

p2 = figure(plot_width=400, plot_height=400, tools=[BoxSelectTool(), TapTool()], title="Components of Y")
p2.circle('x2', 'y2', source=source, size=10, color="firebrick", alpha=0.5, selection_color="orange", nonselection_alpha=0.1)

# JavaScript to link selections
code = '''
if (cb_obj.indices.length > 0) {
    source.selected.indices = cb_obj.indices;
} else {
    source.selected.indices = [];
}
'''

# Attach the callback to both plots
callback = CustomJS(args=dict(source=source), code=code)
source.selected.js_on_change('indices', callback)

# Arrange plots in a row
layout = row(p1, p2)

# Output to static HTML file
output_file("linked_scatter_with_region_select.html")

# Show results
show(layout)

