import os
from os.path import join
import sys

from typing import List, Union, Any, Set, Dict, Tuple, Optional
from functools import partial
from datetime import datetime

import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch import FloatTensor, LongTensor, Tensor
import torch_geometric

from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx

# from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import networkx.algorithms.isomorphism as iso
from collections import Counter

from torch_geometric.nn.conv import MessagePassing
from lib import graph_nns


# from rdkit import Chem, RDLogger
# from rdkit.Chem import Draw, AllChem
# from IPython.display import display

# from joblib import Parallel, delayed

# from lib import utilities
# from lib import featurizers as feat
# from lib.featurizers import AtomFeaturizer, BondFeaturizer

##### From https://colab.research.google.com/drive/1KdUJTEotQ4-vKgOCONV2t2kgIciMTBsD?usp=sharing#scrollTo=X4XKShnmX-Ft

#### CountingGraph Class
# The CountingGraph class is a subclass of NetworkX's Graph class which makes counting subgraphs especially easy.
# It defines a useful neighborhood lookup function as well as hash and equality functions that make it possible to use
# subgraphs as keys in Python dictionaries and easily check for equality, both of which are needed for counting subgraphs.


class CountingGraph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(CountingGraph, self).__init__(*args, **kwargs)

        self._hash = None
        self._neighbors_set: Dict[int, Set[int]] = {}

        # Convert features from list to tuple for hashing
        for node, node_data in self.nodes(data=True):
            node_data["x"] = tuple(node_data["x"])

        for node_1, node_2, edge_data in self.edges(data=True):
            edge_data["edge_attr"] = tuple(edge_data["edge_attr"])

    def neighbors_set(self, node: int) -> Set[int]:
        """Gets a set of neighboring nodes for a given node."""
        if node not in self._neighbors_set:
            self._neighbors_set[node] = set(self.neighbors(node))

        return self._neighbors_set[node]

    def __hash__(self) -> int:
        """Gets the hash of the graph based on node features and degrees.

        Can be used to quickly check potential equality.
        If hashes do not match, graphs are not equal.
        If hashes match, graphs may or may not be equal (call __eq__ to check).
        """
        if self._hash is None:
            self._hash = hash(
                " | ".join(
                    sorted(
                        f"{x} {len(self.neighbors_set(node))}"  # node features and degree
                        for node, x in self.nodes(data="x")
                    )
                )
            )

        return self._hash

    def __eq__(self, other: Any) -> bool:
        """Checks equality between this graph and another as defined by graph isomorphism."""
        if not isinstance(other, CountingGraph):
            return False

        if hash(self) != hash(other):
            return False

        return nx.is_isomorphic(
            self, other, node_match=iso.categorical_node_match(["x"], [None])
        )


def get_exclusive_neighborhood(
    graph: CountingGraph, node: int, subgraph_nodes: Set[int]
) -> Set[int]:
    """Gets the exclusive neighborhood of a node in a graph with respect to a subgraph.

    The exclusive neighborhood of a node consists of all neighbors of the node that are neither in
    the subgraph nor in the neighbors of any of the nodes in the subgraph.

    :param graph: The graph containing the node and subgraph.
    :param node: The node whose exclusive neighborhood should be computed.
    :param subgraph_nodes: The subgraph with respect to which the exclusive neighborhood should be computed.
    :return: The exclusive neighborhood (set of nodes) of the node in the graph with respect to the subgraph.
    """
    subgraph_neighborhood = set.union(
        *[graph.neighbors_set(subgraph_node) for subgraph_node in subgraph_nodes]
    )
    node_neighborhood = graph.neighbors_set(node)
    exclusive_neighborhood = node_neighborhood - subgraph_neighborhood - subgraph_nodes

    return exclusive_neighborhood


def extend_subgraph(
    graph: CountingGraph,
    subgraph_nodes: Set[int],
    extension_nodes: Set[int],
    first_node: int,
    size: int,
) -> List[CountingGraph]:
    """Extends a subgraph by adding adjacent nodes until subgraphs of the correct size are found.

    :param graph: The graph whose induced subgraphs will be enumerated.
    :param subgraph_nodes: The set of nodes in the current subgraph.
    :param extension_nodes: The set of nodes that can be added to extend the subgraph.
    :param first_node: The first node of the subgraph.
    :param size: The desired size of the subgraphs.
    :return: A list of subgraphs of the desired size found by extending the current subgraph.
    """
    # If we have a subgraph of the correct size, return it
    if len(subgraph_nodes) == size:
        return [graph.subgraph(subgraph_nodes)]

    # Extend the current subgraph by adding a node and then recurse
    subgraphs = []
    for i, extension_node in enumerate(list(extension_nodes)):
        # Choose an adjacent node to add to the subgraph
        extension_nodes.remove(extension_node)

        # Get the new list of adjacent nodes
        new_extension_nodes = extension_nodes | {
            neighbor_node
            for neighbor_node in get_exclusive_neighborhood(
                graph=graph, node=extension_node, subgraph_nodes=subgraph_nodes
            )
            if neighbor_node > first_node
        }

        # Extend the subgraph and get all resulting subgraphs of the correct size
        subgraphs += extend_subgraph(
            graph=graph,
            subgraph_nodes=subgraph_nodes | {extension_node},
            extension_nodes=new_extension_nodes,
            first_node=first_node,
            size=size,
        )

    return subgraphs


#### From https://colab.research.google.com/drive/1KdUJTEotQ4-vKgOCONV2t2kgIciMTBsD?usp=sharing#scrollTo=fwUb_ie6BwzD

#### Subgraph Enumeration
#### Here we define functions that implement the ESU algorithm to efficiently enumerate all subgraphs of size k in a graph.


def enumerate_subgraphs(graph: CountingGraph, size: int) -> List[CountingGraph]:
    """Enumerate all induced subgraphs of a given size from a graph.

    :param graph: The graph whose induced subgraphs will be enumerated.
    :param size: The size of the subgraphs to extract.
    :return: A list of all subgraphs of the provided size.
    """
    # Get the nodes of the graph
    nodes = np.array(graph.nodes)

    # Enumerate all subgraphs of the desired ize
    subgraphs = []
    for node in nodes:
        extension_nodes = {
            neighbor_node
            for neighbor_node in graph.neighbors_set(node)
            if neighbor_node > node
        }

        subgraphs += extend_subgraph(
            graph=graph,
            subgraph_nodes={node},
            extension_nodes=extension_nodes,
            first_node=node,
            size=size,
        )

    return subgraphs


#### Subgraph Count Vector
#### Here we define a function that converts a dictionary of subgraph counts to a vector. This vector of subgraph counts is later used as a feature vector representing the graph.


def build_subgraph_vector(
    subgraph_counts: Counter, subgraph_to_index: Dict[CountingGraph, int]
) -> np.ndarray:
    """Builds a vector of the subgraph counts of a graph."""
    subgraph_vector = np.zeros(len(subgraph_to_index))

    for subgraph, count in subgraph_counts.items():
        if subgraph in subgraph_to_index:
            subgraph_vector[subgraph_to_index[subgraph]] = count

    return subgraph_vector


#### Count Subgraphs
#### Now for each graph, we count all the subgraphs in the graph of size k = 5 and create a feature vector based on these counts. Note that the feature vector only includes subgraphs that appear in the training set.

# def count_subgraphs_from_dataset(train_dataset, test_dataset, subgraph_size:int = 5):

#     subgraph_to_index = index_to_subgraph = None
#     train_dataset_subgraph, test_dataset_subgraph = None, None

#     train_dataset_subgraph_X, test_dataset_subgraph_X, train_dataset_subgraph_Y, test_dataset_subgraph_Y = None, None, None, None

#     for split in ['train', 'test']:
#         ds = eval(f'{split}_dataset')
#         print(f'\nDataset with {len(ds)} graphs.')
#         nx_dataset = [to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True) for data in tqdm(ds)]
#         cg_dataset = [CountingGraph(graph) for graph in tqdm(nx_dataset)]
#         subgraph_counts_list = [Counter(enumerate_subgraphs(graph=graph, size=subgraph_size)) for graph in tqdm(cg_dataset)]

#         if split == 'train':
#             unique_subgraphs = set.union(*[set(subgraph_counts) for subgraph_counts in tqdm(subgraph_counts_list)])
#             subgraph_to_index = {subgraph: index for index, subgraph in enumerate(unique_subgraphs)}
#             index_to_subgraph = dict(enumerate(unique_subgraphs))
#             print(f'\nNumber of unique train subgraphs of size {subgraph_size} = {len(unique_subgraphs):,}')

#             train_dataset_subgraph_X = np.array([build_subgraph_vector(subgraph_counts, subgraph_to_index) for subgraph_counts in tqdm(subgraph_counts_list)])
#             train_dataset_subgraph_Y = np.array([data.y[0].item() for data in ds])
#         else:
#             test_dataset_subgraph_X = np.array([build_subgraph_vector(subgraph_counts, subgraph_to_index) for subgraph_counts in tqdm(subgraph_counts_list)])
#             test_dataset_subgraph_Y = np.array([data.y[0].item() for data in ds])

#     return train_dataset_subgraph_X, test_dataset_subgraph_X, train_dataset_subgraph_Y, test_dataset_subgraph_Y, subgraph_to_index, index_to_subgraph


def count_subgraphs_from_train_dataset(
    dataset, is_train: bool = False, subgraph_size: int = 5
):
    subgraph_to_index = index_to_subgraph = None
    dataset_subgraph = None

    dataset_subgraph_X, dataset_subgraph_Y = None, None

    print(f"\nDataset with {len(dataset)} graphs.")
    nx_dataset = [
        to_networkx(
            data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True
        )
        for data in tqdm(dataset)
    ]
    # print(len(nx_dataset))
    cg_dataset = [CountingGraph(graph) for graph in tqdm(nx_dataset)]
    # print(len(cg_dataset))
    subgraph_counts_list = [
        Counter(enumerate_subgraphs(graph=graph, size=subgraph_size))
        for graph in tqdm(cg_dataset)
    ]
    print("subgraph_counts_list", len(subgraph_counts_list))

    unique_subgraphs = set.union(
        *[set(subgraph_counts) for subgraph_counts in tqdm(subgraph_counts_list)]
    )
    subgraph_to_index = {
        subgraph: index for index, subgraph in enumerate(unique_subgraphs)
    }
    index_to_subgraph = dict(enumerate(unique_subgraphs))
    print(
        f"\nNumber of unique train subgraphs of size {subgraph_size} = {len(unique_subgraphs):,}"
    )

    dataset_subgraph_X = np.array(
        [
            build_subgraph_vector(subgraph_counts, subgraph_to_index)
            for subgraph_counts in tqdm(subgraph_counts_list)
        ]
    )
    dataset_subgraph_Y = np.array([data.y[0].item() for data in dataset])

    return dataset_subgraph_X, dataset_subgraph_Y, subgraph_to_index, index_to_subgraph


def count_subgraphs_from_test_dataset(
    dataset, subgraph_to_index, is_train: bool = False, subgraph_size: int = 5
):
    dataset_subgraph = None

    dataset_subgraph_X, dataset_subgraph_Y = None, None

    print(f"\nDataset with {len(dataset)} graphs.")
    nx_dataset = [
        to_networkx(
            data, node_attrs=["x"], edge_attrs=["edge_attr"], to_undirected=True
        )
        for data in tqdm(dataset)
    ]
    print(len(nx_dataset))
    cg_dataset = [CountingGraph(graph) for graph in tqdm(nx_dataset)]
    print(len(cg_dataset))
    subgraph_counts_list = [
        Counter(enumerate_subgraphs(graph=graph, size=subgraph_size))
        for graph in tqdm(cg_dataset)
    ]
    print("subgraph_counts_list", len(subgraph_counts_list))

    if not subgraph_to_index is None:
        dataset_subgraph_X = np.array(
            [
                build_subgraph_vector(subgraph_counts, subgraph_to_index)
                for subgraph_counts in tqdm(subgraph_counts_list)
            ]
        )
        dataset_subgraph_Y = np.array([data.y[0].item() for data in dataset])
    else:
        raise ValueError(
            "subgraph_to_index must be non-null for counting subgraphs test dataset. subgraph_to_index is usually obtained from counting subgraphs from train dataset."
        )

    return dataset_subgraph_X, dataset_subgraph_Y


class MCTSNode(object):
    """A node in a Monte Carlo Tree Search representing a subgraph."""

    def __init__(
        self,
        coalition: Tuple[int, ...],
        data: Data,
        ori_graph: nx.Graph,
        c_puct: float,
        W: float = 0,
        N: int = 0,
        P: float = 0,
    ) -> None:
        """Initializes the MCTSNode object.

        :param coalition: A tuple of the nodes in the subgraph represented by this MCTSNode.
        :param data: The full graph.
        :param ori_graph: The original graph in NetworkX format.
        :param W: The sum of the node value.
        :param N: The number of times of arrival at this node.
        :param P: The property score (reward) of this node.
        :param c_puct: The hyperparameter that encourages exploration.
        """
        self.coalition = coalition
        self.data = data
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.W = W
        self.N = N
        self.P = P
        self.children: List[MCTSNode] = []

    def Q(self) -> float:
        """Value that encourages exploitation of nodes with high reward."""
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        """Value that encourages exploration of nodes with few visits."""
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def size(self) -> int:
        """Returns the number of nodes in the subgraph represented by the MCTSNode."""
        return len(self.coalition)


def gnn_score(coalition: Tuple[int, ...], data: Data, model: torch.nn.Module) -> float:
    """Computes the GNN score of the subgraph with the selected coalition of nodes.

    :param coalition: A list of indices of the nodes to retain in the induced subgraph.
    :param data: A data object containing the full graph.
    :param model: The GNN model to use to compute the score.
    :return: The score of the GNN model applied to the subgraph induced by the provided coalition of nodes.
    """
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    node_mask[list(coalition)] = 1

    # print("coalition", coalition)
    # print("node_mask", node_mask)

    row, col = data.edge_index
    # print("row=", row)
    # print("col=", col)
    # print("node_mask[row]=", node_mask[row])
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    # print(f"edge_mask ({edge_mask.shape})", edge_mask)

    mask_edge_index = data.edge_index[:, edge_mask]
    # print("mask_edge_mask", mask_edge_index)

    mask_global_feats = data.to_dict().get('global_feats', None)





    logits=None

    # if not mask_edge_attr is None:
    if model.can_use_edge_attr:
        mask_edge_attr = data.to_dict().get('edge_attr', None)
        if not mask_edge_attr is None:
            # print(f"mask_edge_attr ({mask_edge_attr.shape})=",mask_edge_attr)
            mask_edge_attr =  mask_edge_attr[edge_mask, :]


        mask_data = Data(x=data.x, edge_index=mask_edge_index, global_feats=mask_global_feats,
                            edge_attr=mask_edge_attr)

        mask_data = Batch.from_data_list([mask_data])

        # print("mask_edge_attr", mask_data.edge_attr.shape)
        logits = model(
            x=mask_data.x, edge_index=mask_data.edge_index, batch=mask_data.batch,
            global_feats = mask_data.to_dict().get('global_feats', None), edge_attr=mask_data.edge_attr
        )
    else:      
        mask_data = Data(x=data.x, edge_index=mask_edge_index, global_feats=mask_global_feats)

        mask_data = Batch.from_data_list([mask_data])

        logits = model(
            x=mask_data.x, edge_index=mask_data.edge_index, batch=mask_data.batch,
            global_feats = mask_data.to_dict().get('global_feats', None)
        )        
    score = torch.sigmoid(logits).item()

    return score


def get_best_mcts_node(results: List[MCTSNode], max_nodes: int) -> MCTSNode:
    """Get the MCTSNode with the highest reward (and smallest graph if tied) that has at most max_nodes nodes.

    :param results: A list of MCTSNodes.
    :param max_nodes: The maximum number of nodes allowed in a subgraph represented by an MCTSNode.
    :return: The MCTSNode with the highest reward (and smallest graph if tied) that has at most max_nodes nodes.
    """
    # Filter subgraphs to only include those with at most max_nodes nodes
    results = [result for result in results if result.size <= max_nodes]

    # Check that there exists a subgraph with at most max_nodes nodes
    if len(results) == 0:
        raise ValueError(f"All subgraphs have more than {max_nodes} nodes.")

    # Sort subgraphs by size in case of a tie (max picks the first one it sees, so the smaller one)
    results = sorted(results, key=lambda result: result.size)

    # Find the subgraph with the highest reward and break ties by preferring a smaller graph
    best_result = max(results, key=lambda result: (result.P, -result.size))

    return best_result


class MCTS(object):
    """An object which runs Monte Carlo Tree Search to find optimal subgraphs of a graph."""

    def __init__(
        self,
        x: FloatTensor,
        edge_index: LongTensor,
        model: torch.nn.Module,
        num_hops: int,
        n_rollout: int,
        min_nodes: int,
        c_puct: float,
        num_expand_nodes: int,
        high2low: bool,
        global_feats: LongTensor = None,
        edge_attr: LongTensor = None

    ) -> None:
        """Creates the Monte Carlo Tree Search (MCTS) object.

        :param x: Input node features.
        :param edge_index: The edge indices.
        :param model: The GNN model to explain.
        :param num_hops: The number of hops to extract the neighborhood of target node.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_nodes: Number of graph nodes of the leaf node in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of nodes to expand when extending the child nodes in the search tree.
        :param high2low: Whether to expand children nodes from high degree to low degree
                         when extending the child nodes in the search tree.
        """
        self.x = x
        self.edge_index = edge_index
        self.global_feats = global_feats
        self.edge_attr = edge_attr
        self.model = model
        self.num_hops = num_hops
        self.data = Data(x=self.x, edge_index=self.edge_index, 
                            global_feats=self.global_feats, edge_attr=self.edge_attr)
        self.graph = to_networkx(
            Data(x=self.x, edge_index=remove_self_loops(self.edge_index)[0]),
            to_undirected=True,
        )
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

        self.root_coalition = tuple(range(self.num_nodes))
        self.MCTSNodeClass = partial(
            MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct
        )
        self.root = self.MCTSNodeClass(coalition=self.root_coalition)
        self.state_map = {self.root.coalition: self.root}

    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        """Performs a Monte Carlo Tree Search rollout.

        :param tree_node: An MCTSNode representing the root of the MCTS search.
        :return: The value (reward) of the rollout.
        """
        if len(tree_node.coalition) <= self.min_nodes:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            # Maintain a set of all the coalitions added as children of the tree
            tree_children_coalitions = set()

            # Get subgraph induced by the tree
            tree_subgraph = self.graph.subgraph(tree_node.coalition)

            # Get nodes to try expanding
            all_nodes = sorted(
                tree_subgraph.nodes,
                key=lambda node: tree_subgraph.degree[node],
                reverse=self.high2low,
            )
            all_nodes_set = set(all_nodes)

            expand_nodes = all_nodes[: self.num_expand_nodes]

            # For each node, prune it and get the remaining subgraph (only keep the largest connected component)
            for expand_node in expand_nodes:
                subgraph_coalition = all_nodes_set - {expand_node}

                subgraphs = (
                    self.graph.subgraph(connected_component)
                    for connected_component in nx.connected_components(
                        self.graph.subgraph(subgraph_coalition)
                    )
                )

                subgraph = max(
                    subgraphs, key=lambda subgraph: subgraph.number_of_nodes()
                )

                new_coalition = tuple(sorted(subgraph.nodes()))

                # Check the state map and merge with an existing subgraph if available
                new_node = self.state_map.setdefault(
                    new_coalition, self.MCTSNodeClass(coalition=new_coalition)
                )

                # Add the subgraph to the children of the tree
                if new_coalition not in tree_children_coalitions:
                    tree_node.children.append(new_node)
                    tree_children_coalitions.add(new_coalition)

            # For each child in the tree, compute its reward using the GNN
            for child in tree_node.children:
                if child.P == 0:
                    # print("child.data", child.data)
                    child.P = gnn_score(
                        coalition=child.coalition, data=child.data, model=self.model
                    )

        # Select the best child node and unroll it
        sum_count = sum(child.N for child in tree_node.children)
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(n=sum_count))
        v = self.mcts_rollout(tree_node=selected_node)
        selected_node.W += v
        selected_node.N += 1

        return v

    def run_mcts(self) -> List[MCTSNode]:
        """Runs the Monte Carlo Tree search.

        :return: A list of MCTSNode objects representing subgraph explanations sorted from highest to
                 smallest reward (for ties, the smaller graph is first).
        """
        for _ in trange(self.n_rollout):
            self.mcts_rollout(tree_node=self.root)

        explanations = [node for _, node in self.state_map.items()]

        # Sort by highest reward and break ties by preferring a smaller graph
        explanations = sorted(explanations, key=lambda x: (x.P, -x.size), reverse=True)

        return explanations


class SubgraphX(object):
    """An object which contains methods to explain a GNN's prediction on a graph in terms of subgraphs."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_hops: Optional[int] = None,
        n_rollout: int = 20,
        min_nodes: int = 5,
        c_puct: float = 10.0,
        num_expand_nodes: int = 14,
        high2low: bool = False,
    ) -> None:
        """Initializes the SubgraphX object.

        :param model: The GNN model to explain.
        :param num_hops: The number of hops to extract the neighborhood of target node.
                         If None, uses the number of MessagePassing layers in the model.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_nodes: Number of graph nodes of the leaf node in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of nodes to expand when extending the child nodes in the search tree.
        :param high2low: Whether to expand children nodes from high degree to low degree
                         when extending the child nodes in the search tree.
        """
        self.model = model
        self.model.eval()
        self.num_hops = num_hops

        if self.num_hops is None:
            self.num_hops = sum(
                isinstance(module, MessagePassing) for module in self.model.modules()
            )

        # MCTS hyperparameters
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

    def explain(self, x: Tensor, edge_index: Tensor, max_nodes: int, global_feats: Tensor=None, edge_attr: Tensor=None) -> MCTSNode:
        """Explain the GNN behavior for the graph using the SubgraphX method.

        :param x: Node feature matrix with shape [num_nodes, dim_node_feature].
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges].
        :param max_nodes: The maximum number of nodes in the final explanation results.
        :return: The MCTSNode corresponding to the subgraph that best explains the model's prediction on the graph
                 (the smallest graph that has the highest reward such that the subgraph has at most max_nodes nodes).
        """
        # Create an MCTS object with the provided graph
        mcts = MCTS(
            x=x,
            edge_index=edge_index,
            global_feats=global_feats,
            edge_attr=edge_attr,
            model=self.model,
            num_hops=self.num_hops,
            n_rollout=self.n_rollout,
            min_nodes=self.min_nodes,
            c_puct=self.c_puct,
            num_expand_nodes=self.num_expand_nodes,
            high2low=self.high2low,
        )

        # Run the MCTS search
        mcts_nodes = mcts.run_mcts()

        # Select the MCTSNode that contains the smallest subgraph that has the highest reward
        # such that the subgraph has at most max_nodes nodes
        best_mcts_node = get_best_mcts_node(mcts_nodes, max_nodes=max_nodes)

        return best_mcts_node
