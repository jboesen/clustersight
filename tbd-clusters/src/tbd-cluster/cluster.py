"""
A module for understanding clustered data
"""
from math import sqrt, ceil
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from plotly.graph_objs import FigureWidget, Scatter, Table
import plotly.offline as po
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import VBox
po.init_notebook_mode()

# Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# run pca on the iris dataset

def create_lasso(data=df):
    """
    Input: Datafame
    Output: Plotly FigureWidget with lasso select tool
    """
    pca = PCA(n_components=2)
    pca.fit(df)
    pca_df = pd.DataFrame(pca.transform(df), columns=['x', 'y'])
    df['x'] = pca_df['x']
    df['y'] = pca_df['y']

    f = FigureWidget([Scatter(y = df["x"], x = df["y"], mode = 'markers')])
    scatter = f.data[0]
    df.dropna()

    N = len(df)
    scatter.marker.opacity = 0.5

    # Create a table FigureWidget that updates on selection from points in the scatter plot of f
    t = FigureWidget([Table(
        header=dict(values=df.columns,
                    fill = dict(color='#C2D4FF'),
                    align = ['left'] * 5),

        cells=dict(values=[df[col] for col in df.columns],

                fill = dict(color='#F5F8FF'),
                align = ['left'] * 5
                ))])
    def selection_fn(trace,points,selector):
        t.data[0].cells.values = [df.loc[points.point_inds][col] for col in df.columns]
    scatter.on_selection(selection_fn)

    #iplot({data : scatter.on_selection(selection_fn)})

    # Put everything together
    VBox((f, t))
# create histogram
def create_histograms(df=df, exclude_cols=['target', 'x', 'y']):
    """
    Input: Dataframe, list of columns to exclude
    Output: Plotly FigureWidget with histograms of each column
    """
    curr_df = df.drop(exclude_cols, axis=1)
    print(curr_df.columns)
    r = int(sqrt(len(curr_df.columns)))
    c = ceil(len(curr_df.columns) / r)
    print(r, c)
    fig = make_subplots(rows=r, cols=c)
    col_num =0
    for i in range(1, r+1):
        for j in range(1, c+1):
            print(i, j)
            fig.add_trace(go.Histogram(x=curr_df[curr_df.columns[col_num]], name=curr_df.columns[col_num]), row=i, col=j) 
            fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.2, showarrow=False,
                   text=f"<b>{curr_df.columns[col_num]}</b>", row=i, col=j)
            col_num += 1
    fig.show()
create_histograms()
