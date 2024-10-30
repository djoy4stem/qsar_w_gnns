import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns

import networkx as nx

from typing import List, Union, Tuple


#######################################
#             VISUALIZATION           #
#######################################
def plots_train_val_metrics(
    train_losses: List[float],
    val_scores: List[float],
    val_losses: List[float] = None,
    figsize: Tuple = (10, 7),
    image_pathname: str = None,
    val_score_name: str = None,
):
    plt.figure(figsize=figsize)
    plt.plot(train_losses, color="orange", label="train loss")
    if not val_losses is None:
        plt.plot(val_losses, color="red", label="val. loss")
    val_score_label = (
        "val. score" if val_score_name is None else f"val. score ({val_score_name})"
    )

    plt.plot(val_scores, color="green", label=val_score_label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Score")
    plt.legend()
    if not image_pathname is None:
        plt.savefig(image_pathname)
    plt.show()


def visualize_colored_graph(edge_index, edge_mask, data, threshold=0.5):
    # Build graph from edge_index
    G = nx.Graph()
    edge_list = edge_index.t().tolist()  # Convert edge_index to list of edges

    # Remove self-loops (edges where source and target nodes are the same)
    edge_list = [edge for edge in edge_list if edge[0] != edge[1]]

    G.add_edges_from(edge_list)

    # Convert edge_mask to numpy and resize if necessary
    edge_mask = edge_mask.cpu().detach().numpy()  # Convert tensor to numpy
    if len(edge_mask) != len(edge_list):
        edge_mask = edge_mask[: len(edge_list)]

    important_edges = edge_mask > threshold

    # Normalize the edge_mask for coloring
    edge_mask_normalized = edge_mask / edge_mask.max()  # Normalize to [0, 1]

    # Create the list of edge colors using colormap
    cmap = plt.cm.Reds  # Colormap to use
    edge_colors = cmap(edge_mask_normalized)  # Get colors for edges

    # Filter the edges based on importance (threshold)
    edges_to_draw = np.array(edge_list)[important_edges]
    edge_colors_to_draw = edge_colors[important_edges]

    # Get node positions
    pos = nx.spring_layout(G)  # Position the graph layout
    plt.figure(figsize=(8, 6))  # Set figure size

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)

    # Draw edges with corresponding colors
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_to_draw,
        edge_color=edge_colors_to_draw,
        edge_cmap=cmap,
        width=2,
        edge_vmin=0,
        edge_vmax=1,
    )

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Add colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # Set the array to use for color mapping

    # Create a new Axes for colorbar
    ax_colorbar = plt.axes(
        [0.95, 0.12, 0.03, 0.75]
    )  # Adjust the position and size of the colorbar
    plt.colorbar(sm, cax=ax_colorbar)  # Create the colorbar

    plt.title("Graph with Edge Importance")
    plt.show()


def reduce_and_plot(data, actuals=None, method='umap', n_components=2, figsize=(10, 8), random_state=42, **kwargs):
    """
    Reduces dimensionality of data using UMAP or t-SNE and plots it.

    Parameters:
    - data: numpy array or tensor of shape (n_samples, n_features)
    - actuals: array-like of shape (n_samples,), optional
      Labels for coloring different classes in the plot
    - method: str, 'umap' or 'tsne', specifies the dimensionality reduction method
    - n_components: int, number of dimensions to reduce to (typically 2 for visualization)
    - kwargs: additional arguments passed to UMAP or t-SNE

    Returns:
    - embedding: numpy array of shape (n_samples, n_components), the reduced data
    """
    if method == 'umap':
        ## To ensure deterministic results, make sure to set the random state and transform_seed variables
        # According to the 2022 discussion below below, they do not have a big impact. But my visual inspection 
        # on datasets in 2024 show they do. The issue has been close since then.
        # https://github.com/lmcinnes/umap/issues/158#issuecomment-472220347
        # Also check UMAP Reproducibility at https://umap-learn.readthedocs.io/en/latest/reproducibility.html

        reducer = UMAP(n_components=n_components, random_state=random_state, transform_seed=random_state, **kwargs)

    
    elif method == 'tsne':
        ## 
        # 1) Mastering t-SNE(t-distributed stochastic neighbor embedding):
        #   https://medium.com/@sachinsoni600517/mastering-t-sne-t-distributed-stochastic-neighbor-embedding-0e365ee898ea
        # 2) https://www.datacamp.com/tutorial/introduction-t-sne
        # 3) https://distill.pub/2016/misread-tsne/
        
        reducer = TSNE(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError("Invalid method. Choose either 'umap' or 'tsne'.")

    # Fit and transform the data
    if method == 'tsne' and isinstance(data, (List, np.ndarray)) and isinstance(data[0], (List, np.ndarray)):
            embedding = reducer.fit_transform(pd.DataFrame(data))
    else:
        embedding = reducer.fit_transform(data)

    # Plotting the results
    plt.figure(figsize=figsize)
    if actuals is not None:
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=actuals, palette="viridis", s=60, alpha=0.8)
        plt.legend(title="Classes")
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=60, alpha=0.8)
    plt.title(f'{method.upper()} projection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    return embedding


def plot_embeddings(model, data, actuals=None, method='umap', n_components=2, **kwargs):

    h = model(data)
    reduce_and_plot(data=data, actuals=actuals, method=methods, n_components=n_components, **kwargs)