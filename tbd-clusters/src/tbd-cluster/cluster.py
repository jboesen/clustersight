import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from plotly.graph_objs import FigureWidget, Scatter, Table
import plotly.offline as po
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
