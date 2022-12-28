import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA

# https://plotly.com/python/v3/selection-events/
# Load the iris dataset into a Pandas DataFrame
iris = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/iris.csv")

# Use PCA to compress the dataset down to two dimensions
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris.iloc[:, 0:4])

# Create a scatterplot using Plotly Express, setting the `x` and `y` values to the first and second dimensions of the PCA-transformed data
fig = px.scatter(iris, x=iris_pca[:, 0], y=iris_pca[:, 1])

# Add the lasso select tool to the scatterplot
fig.update_layout(dragmode='lasso')

# Define a function that takes a list of selected datapoints as input and performs some operation on them
def process_selected_points(trace, points, selector):
    # do something with the selected points
    print(points.point_inds)

# Use the `plotly.graph_objects.FigureWidget.on_selection` method to register a callback function that is called when the user makes a selection using the lasso tool
fig.data[0].on_selection(process_selected_points)

# Display the plot
fig.show()
