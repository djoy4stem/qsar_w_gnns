import numpy as np
import networkx as nx
from typing import Optional, List, Set, Union, Tuple

from lib import predictions, graph_utils
from ipywidgets import interact
import ipywidgets as widgets

from typing import List, Union
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx
from torch_geometric.nn.conv import MessagePassing


## Generated color code https://g.co/kgs/FuoNWHq


NODE_TO_COLOR_DICT = {
    "H": "#EDF2EF",
    "C": "#EDE6D8",
    "N": "#4970C6",
    "O": "#FF5357",
    "F": "#D446DB",
    "P": "brown",
    "S": "#F2F05C",
    "Cl": "#7AF246",
    "Br": "#C9C720",
    "I": "#AD49C6",
    "Se": "#807E04",
    "B": "#6B9C4F",
    "Si": "#A8A73D",
    "Na": "#3D7AA8",
    "Mg": "#3DA86D",
    "unk": "#C66F49",
}


# Class for explaining predictions using GNNExplainer
# The code was adapted from examples provided by thse resources.
# 1. Why should I trust my Graph Neural Network: https://medium.com/stanford-cs224w/why-should-i-trust-my-graph-neural-network-4d964052bd85
# 2. PyTorch Geometric GNNExplainer example: "https://github.com/pyg-team/pytorch_geometric/blob/master/examples/explain/gnn_explainer.py"


class GNNExplainerModule:
    def __init__(
        self,
        gnn_predictor: predictions.GNNPredictor,
        node_dict: dict,
        num_epochs: int = 100,
        lr: float = 0.05,
        node_mask_type: str = "attributes",
        edge_mask_type: str = "object",
        explanation_type: str = "model",
        model_config: dict = dict(
            mode="binary_classification", task_level="graph", return_type="raw"
        ),
        node_colors: List[str] = None,
        device: str = "cpu",
    ):
        if not node_colors is None:
            assert len(node_colors) >= len(
                node_dict
            ), "ValueError: The number of provided node colors is lower than the number of node types."
        assert node_dict is None or len(
            node_dict
        ), "ValueError: You must provide a non empty dictioary of nodes."

        self.gnn_predictor = gnn_predictor
        self.node_dict = node_dict
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type
        self.explanation_type = explanation_type
        self.model_config = model_config
        self.explainer = Explainer(
            model=gnn_predictor.model.to(device),
            algorithm=GNNExplainer(epochs=num_epochs, lr=0.05),
            explanation_type=self.explanation_type,
            node_mask_type=self.node_mask_type,
            edge_mask_type=self.edge_mask_type,
            model_config=self.model_config,
        )

        if node_colors is None:
            self.create_node_colors()
        else:
            self.node_colors = node_colors

        self.device = device
        self.subgraph_logred_model = None
        self.subgraph_size = None

        # print("node_dict = " , self.node_dict)
        # print("node_colors = " , self.node_colors)

    def create_node_colors(self):
        colors = []
        for c in self.node_dict.values():
            if c in NODE_TO_COLOR_DICT:
                colors.append(NODE_TO_COLOR_DICT[c])
            else:
                colors[c].append(NODE_TO_COLOR_DICT["unk"])
        self.node_colors = colors

    def visualize_subgraph(
        self,
        graph: nx.Graph,
        node_set: Optional[Set[int]] = None,
        edge_set: Optional[Set[int]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple] = (8,6),
        node_size: int = 300,
        edge_width: int = 6
    ) -> None:
        """Visualizes a subgraph explanation for a graph.

        Note: Only provide subgraph_node_set or subgraph_edge_set, not both.

        Adapted from https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

        :param graph: A NetworkX graph object representing the full graph.
        :param node_set: A set of nodes that induces a subgraph.
        :param edge_set: A set of edges that induces a subgraph.
        :param title: Optional title for the plot.
        """
        if node_set is None:
            node_set = set(graph.nodes())

        if edge_set is None:
            # edge_set = {(n_from, n_to) for (n_from, n_to) in graph.edges() if n_from in node_set and n_to in node_set}
            edge_set = {
                (n_from, n_to)
                for (n_from, n_to) in graph.edges()
                if n_from in node_set and n_to in node_set
            }

        # for node, node_x in graph.nodes(data='x'):
        #     print(f"Node = {node} ------- Node_x = {node_x}")

        ## Here, we focus on the len(node_dict) first elements of x, as they represent the different atom types that we condier here.
        node_idxs = {
            node: node_x[: len(self.node_dict)].index(1.0)
            for node, node_x in graph.nodes(data="x")
        }
        node_labels = {k: self.node_dict[v] for k, v in node_idxs.items()}
        # print("Node IDx", node_idxs.keys())
        # print("Node labels", node_labels)

        colors = [
            self.node_colors[v % len(self.node_colors)] for k, v in node_idxs.items()
        ]

        pos = nx.kamada_kawai_layout(graph)

        nx.draw_networkx_nodes(
            G=graph,
            pos=pos,
            nodelist=list(graph.nodes()),
            node_color=colors,
            node_size=node_size,
        )
        nx.draw_networkx_edges(
            G=graph, pos=pos, width=3, edge_color="gray", arrows=False
        )
        nx.draw_networkx_edges(
            G=graph,
            pos=pos,
            edgelist=list(edge_set),
            width=edge_width,
            edge_color="black",
            arrows=False,
        )
        nx.draw_networkx_labels(G=graph, pos=pos, labels=node_labels)

        if title is not None:
            plt.title(title)

        plt.axis("off")
        plt.show()
        plt.close()

    def visualize_explanation_for_graph(
        self,
        my_data: Data,
        threshold: float = 0.5,
        device: str = "cpu",
        remove_self_loops: bool = True,
        figsize: Optional[Tuple] = (8,6),
        node_size: int = 300,
        edge_width: int = 6
    ) -> None:
        """Visualizes the explanations of GNNExplainer for a graph given a mask threshold."""
        device = device or self.device
        explanation = self.explainer(
            x=my_data.x.to(device),
            edge_index=my_data.edge_index.to(device),
            batch=my_data.to(device).batch,
            global_feats=my_data.to(device).to_dict().get("global_feats", None),
        )

        edge_mask = explanation.edge_mask
        # print("edge_mask = ", edge_mask)

        pred = self.gnn_predictor.predict_from_data_list(
            my_data, device=device
        ).detach()  # .to(device)
        # print(f"pred[0][0] = {pred[0][0]}")

        edge_set = {
            (edge[0].item(), edge[1].item())
            for edge, mask in zip(my_data.edge_index.T, edge_mask)
            if mask > threshold
        }
        graph = to_networkx(
            my_data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=False,
            remove_self_loops=remove_self_loops,
            
        )

        if "y" in my_data:
            # label =
            self.visualize_subgraph(
                graph=graph,
                edge_set=edge_set,
                title=f"GNNExplainer on graph : label = {my_data.y.item()}, pred = {pred[0][0]:.2f}",
                node_size=node_size,
                edge_width=edge_width
            )
        else:
            self.visualize_subgraph(
                graph=graph,
                edge_set=edge_set,
                title=f"GNNExplainer on graph : pred = {pred[0][0]:.2f}",
                node_size=node_size,
                edge_width=edge_width                
            )

    def interactive_explanation_viz_for_graphs(
        self, graphs: List[Data], max_num: int = 25, threshold: float = 0.5, device=None,
                        node_size: int = 300, edge_width: int = 6
    ):
        device = device or self.device
        max_num = min(len(graphs), max_num)

        print("\n### Interactive exploration of GNNExplainer explanations ###\n")

        @interact(
            threshold=widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01),
            graph_idx=widgets.IntSlider(value=0, min=0, max=max_num),
        )
        def interactive_class_explanations(graph_idx: int, threshold: float):
            self.visualize_explanation_for_graph(
                my_data=graphs[graph_idx],
                threshold=threshold,
                device=device,
                remove_self_loops=True,
                node_size=node_size,
                edge_width=edge_width

            )

    ### Predict via Subgraph Counts
    ## Using the feature vectors of subgraph counts defined above, we train a logistic regression model with L1 penalty.
    # The L1 penalty is used to encourage a sparse solution that uses relatively few subgraphs. We apply this model to the test set
    # and evaluate its predictions as well as the number of subgraphs it uses, which corresponds to the number of non-zero coefficients in the model.
    def train_subgraph_based_logreg_predictor(
        self, train_dataset: List[Data], subgraph_size: int = 5
    ):
        (
            train_dataset_subgraph_X,
            train_dataset_subgraph_Y,
            subgraph_to_index,
            index_to_subgraph,
        ) = graph_utils.count_subgraphs_from_train_dataset(
            train_dataset[:], is_train=True, subgraph_size=subgraph_size
        )
        print(
            f"Train: {train_dataset_subgraph_X.shape}, {train_dataset_subgraph_Y.shape}"
        )

        # Create logistic regression model with L1 penalty
        self.subgraph_logred_model = LogisticRegression(
            penalty="l1", C=1.0, solver="liblinear", max_iter=1000, random_state=0
        )

        # Train model
        self.subgraph_logred_model.fit(
            train_dataset_subgraph_X, train_dataset_subgraph_Y
        )

        self.subgraph_size = subgraph_size
        self.subgraph_to_index = subgraph_to_index
        self.index_to_subgraph = index_to_subgraph
        # print("self.index_to_subgraph is None", self.index_to_subgraph is None)
        # print("self.index_to_subgraph", self.index_to_subgraph)

    def evaluate_subgraph_based_logreg_predictor(self, test_dataset: List[Data]):
        (
            test_dataset_subgraph_X,
            test_dataset_subgraph_Y,
        ) = graph_utils.count_subgraphs_from_test_dataset(
            test_dataset[:],
            is_train=False,
            subgraph_size=self.subgraph_size,
            subgraph_to_index=self.subgraph_to_index,
        )
        print(
            f"Test:  {test_dataset_subgraph_X.shape} , {test_dataset_subgraph_Y.shape}"
        )

        # Predict on the test set
        test_preds = self.subgraph_logred_model.predict(test_dataset_subgraph_X)

        # Evaluate test performance
        test_true = test_dataset_subgraph_Y

        auc = roc_auc_score(test_true, test_preds)
        accuracy = accuracy_score(test_true, test_preds)

        print("\nModel stats\n************")
        print(f"Accuracy for subgraphs of size {self.subgraph_size}: {accuracy:.4f}")
        print(f"ROC AUC for subgraphs of size {self.subgraph_size}: {auc:.4f}")
        print(
            f"Number of subgraphs used = {(self.subgraph_logred_model.coef_[0] != 0).sum()} / {len(self.subgraph_to_index)}"
        )

        return auc, accuracy

    def identify_most_important_subgraphs(self):
        assert (
            not self.subgraph_logred_model is None
        ), "A subgraph-based prediction model is not existant, and must be trained. Please train a model first."

        feature_coefficients = self.subgraph_logred_model.coef_[0]
        feature_argsort = np.argsort(np.abs(feature_coefficients))[
            ::-1
        ]  # Sort from highest to lowest importance
        coefficients_sorted = feature_coefficients[feature_argsort]
        subgraphs_sorted = [self.index_to_subgraph[index] for index in feature_argsort]

        # self.subgraphs_sorted = subgraphs_sorted
        # self.coefficients_sorted = coefficients_sorted

        return subgraphs_sorted, coefficients_sorted

    def visualize_top_k_subgraphs(self, k: int = 5, node_size: int = 300, edge_width: int = 6):
        subgraphs_sorted, coefficients_sorted = self.identify_most_important_subgraphs()

        for i in range(k):
            self.visualize_subgraph(
                graph=subgraphs_sorted[i],
                title=f"Subgraph {i + 1} with coefficient = {coefficients_sorted[i]:.2f}",
                node_size=node_size, edge_width=edge_width
            )

    ## This is suited for binary graph classification
    def visualize_subgraphx_explanations(
        self, dataset: List[Data], device: str = "cpu", num_nodes: int = 10, 
        c_puct: float = 10.0,
        num_expand_nodes: int = 14,
        high2low: bool = False,
        node_size: int = 300, edge_width: int = 6
    ):
        subgraphx = graph_utils.SubgraphX(
            model=self.gnn_predictor.model.to(device), min_nodes=num_nodes,
            c_puct=c_puct, num_expand_nodes=num_expand_nodes,
            high2low=high2low
        )

        for i, data in enumerate(dataset):
            graph = to_networkx(
                data,
                node_attrs=["x"],
                edge_attrs=["edge_attr"],
                to_undirected=True,
                remove_self_loops=True,
            )
            subgraph = subgraphx.explain(
                x=data.x.to(device),
                edge_index=data.edge_index.to(device),
                max_nodes=num_nodes,
            )

            if hasattr(data, "y"):
                self.visualize_subgraph(
                    graph=graph,
                    node_set=set(subgraph.coalition),
                    title=f"SubgraphX on graph {i} : label = {data.y.item()}",
                    node_size=node_size, 
                    edge_width=edge_width
                )
            else:
                self.visualize_subgraph(
                    graph=graph,
                    node_set=set(subgraph.coalition),
                    title=f"SubgraphX on graph {i}",
                    node_size=node_size, 
                    edge_width=edge_width
                )

    def interactive_subgraphx_based_explanation_viz_for_graphs(
        self, dataset: List[Data], device: str = "cpu", num_nodes: int = 10, 
        c_puct: float = 10.0,
        num_expand_nodes: int = 14,
        high2low: bool = False,
        max_num: int = 25, threshold: float = 0.5,
        node_size: int = 300, edge_width: int = 6
    ):
        device = device or self.device
        max_num = min(len(dataset), max_num)

        print("\n### Interactive exploration of GNNExplainer explanations ###\n")

        @interact(
            threshold=widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01),
            graph_idx=widgets.IntSlider(value=0, min=0, max=max_num),
        )
        def interactive_subgraphx_based_class_explanations(graph_idx: int, threshold: float):
            self.visualize_subgraphx_explanations(                
                dataset=[dataset[graph_idx]],
                device=device,
                num_nodes=num_nodes,
                c_puct=c_puct,
                num_expand_nodes=num_expand_nodes,
                high2low=high2low,
                node_size=node_size,
                edge_width=edge_width

            )


# ### OLD CODE
# from torch_geometric.explain import Explainer, GNNExplainer
# device = 'cuda:0'
# # model_ = torch.load(f"{DATASET_DIR}/models/dd2_class_gcn_model.pt")
# # model_ = model_.to(device)
# num_epochs=1000
# my_gnn_explainer = Explainer(model=gcn_predictor.model.to(device), algorithm=GNNExplainer(epochs=num_epochs, lr=0.05), explanation_type='model'
#                         , node_mask_type='attributes', edge_mask_type='object'
#                         # , node_mask_type=None, edge_mask_type='object'
#                         , model_config=dict( mode='binary_classification', task_level='graph', return_type='raw', ), )

# ### https://github.com/pyg-team/pytorch_geometric/blob/master/examples/explain/gnn_explainer.py
# ## Make sure to install graphviz , and xdg-utils
# ## sudo apt install graphviz
# ## sudo apt install xdg-utils
# ## If you want the graphviz visualization to be open using a pdf viewer, instead of saving it to file first, install a pdf reader such as evince (sudo apt install evince)

# import matplotlib.pyplot as plt

# # Set a custom figure size (width, height)
# plt.figure(figsize=(14, 10))

# print(len(test_loader.dataset))
# print(test_loader.dataset[1].batch)
# explanations = [explainer(x=test_loader.dataset[i].x, edge_index=test_loader.dataset[i].edge_index, batch=test_loader.dataset[i].batch, global_feats=test_loader.dataset[i].to_dict().get('global_feats',None)) for i in range(5)]
# print(len(explanations))
# explanations[0].visualize_graph(backend="networkx") #, path="/home/djoy2409-wsl/projects/software_development/qsar_w_gnns/data/explanation.pdf"
# explanations[0].visualize_feature_importance(top_k=10)
