import torch
import torch.nn as nn
from torch import cat, tensor
import torch.nn.init as init
from torch_geometric.nn import GCNConv, Linear, GATv2Conv, GINConv
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as Fun
from torch.nn import (
    Linear,
    Sequential,
    BatchNorm1d,
    ReLU,
    LeakyReLU,
    Softmax,
    Sigmoid,
    ModuleList,
    Dropout,
)

from tqdm import tqdm
import numpy as np
from typing import List, Union, Any
from lib import gnn_utils, datasets, featurizers as feat
from lib.featurizers import AtomFeaturizer, BondFeaturizer, MoleculeFeaturizer

import time

_allowable_tasks = [
    "binary_classification",
    "multiclass_classification",
    "regression",
]  # , 'multi_class_classification'
_allowable_pooling_funcs = ["mean", "sum", "max"]





class GCN(nn.Module):
    def __init__(
        self,
        task: str,
        in_channels,
        gnn_hidden_neurons: int = 128,
        gnn_nlayers: int = 2,
        global_fdim: int = None,
        ffn_hidden_neurons: int = 128,
        ffn_nlayers: int = 1,
        out_neurons: int = 1,
        dropout_rate: float = 0.3,
        gpooling_func: str = "mean",
        activation_func: object = torch.nn.ReLU(),
        add_batch_norm: bool = False,
    ):
        super(GCN, self).__init__()

        assert (
            task.lower() in _allowable_tasks
        ), f"The task must be either of the following: {'; '.join(_allowable_tasks)}."
        assert (
            gpooling_func.lower() in _allowable_pooling_funcs
        ), "The global poling function must be either of the following: {'; '.join(_allowable_pooling_funcs)}."

        self.task = task.lower()
        self.in_channels = in_channels
        self.gnn_dim = gnn_hidden_neurons
        self.gnn_nlayers = gnn_nlayers
        self.global_fdim = global_fdim
        self.ffn_dim = ffn_hidden_neurons
        self.ffn_nlayers = ffn_nlayers
        self.out_neurons = out_neurons
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        if isinstance(self.activation_func, str):
            self.activation_func = eval(f"{self.activation_func}()")
        self.add_batch_norm = add_batch_norm

        self.set_global_pooling_func(gpooling_func)

        self.create_layers()
        self.initialize_params()

    def initialize_params(self):
        print("Initializing parameters...")
        for layer_ in self.modules():
            # print(layer_.__class__)
            if isinstance(layer_, GCNConv):
                layer_.reset_parameters()
            if isinstance(layer_, Linear):
                # print(f"\t{layer_.weight.shape}")
                init.xavier_uniform_(layer_.weight)
                init.zeros_(layer_.bias)

    @classmethod
    def from_dict(cls, params: dict):
        print("params = ", params)
        task = params.get("task").lower()
        in_channels = params.get("in_channels")
        gnn_hidden_neurons = params.get("gnn_hidden_neurons", 128)
        gnn_nlayers = params.get("gnn_nlayers", 2)
        global_fdim = params.get("global_fdim", None)
        ffn_hidden_neurons = params.get("ffn_hidden_neurons", 128)
        ffn_nlayers = params.get("ffn_nlayers", 1)
        out_neurons = params.get("out_neurons", 1)
        dropout_rate = params.get("dropout_rate", 0.3)
        activation_func = params.get("activation_func", torch.nn.ReLU())
        add_batch_norm = params.get("add_batch_norm", False)

        gpooling_func = params.get("gpooling_func", "mean")

        return cls(
            task=task,
            in_channels=in_channels,
            gnn_hidden_neurons=gnn_hidden_neurons,
            global_fdim=global_fdim,
            gnn_nlayers=gnn_nlayers,
            ffn_hidden_neurons=ffn_hidden_neurons,
            ffn_nlayers=ffn_nlayers,
            out_neurons=out_neurons,
            dropout_rate=dropout_rate,
            activation_func=activation_func,
            gpooling_func=gpooling_func,
            add_batch_norm=add_batch_norm,
        )

    def set_global_pooling_func(self, gpooling_func):
        if gpooling_func == "mean":
            self.global_pooling = global_mean_pool
        elif gpooling_func == "sum":
            self.global_pooling = global_add_pool
        elif gpooling_func == "max":
            self.global_pooling = global_max_pool

    def create_layers(self):
        # print('self.in_channels', self.in_channels)
        # print('self.gnn_dim', self.gnn_dim)
        if self.task in ["binary_classification"]:
            self.classification_out_layer = Sigmoid()
        elif self.task in ["multiclass_classification"]:
            self.classification_out_layer = Softmax(dim=1)

        self.conv1 = GCNConv(self.in_channels, self.gnn_dim)

        ## If there are more than one graph convolutional layers.
        self.convs = ModuleList(
            [GCNConv(self.gnn_dim, self.gnn_dim) for x in range(self.gnn_nlayers - 1)]
        )

        ## Build the Feed-forward neural network

        if self.ffn_nlayers == 1:

            if not self.global_fdim is None:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim + self.global_fdim, self.out_neurons),
                ]
            else:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim, self.out_neurons),
                ]

            if self.add_batch_norm:
                ffn.append(BatchNorm1d(self.out_neurons))

            self.ffn = Sequential(*ffn)

        elif self.ffn_nlayers > 1:
            # ffn = [
            #         Dropout(p=self.dropout_rate, inplace=False),
            #         Linear(self.gnn_dim, self.ffn_dim)

            #     ]
            if not self.global_fdim is None:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim + self.global_fdim, self.ffn_dim),
                ]
            else:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim, self.ffn_dim),
                ]

            if self.add_batch_norm:
                ffn.append(BatchNorm1d(self.ffn_dim))
                # if not self.global_fdim is None:
                #     ffn.append(BatchNorm1d(self.ffn_dim + self.global_fdim))
                # else:
                #     ffn.append(BatchNorm1d(self.ffn_dim))

            for x in range(self.ffn_nlayers - 2):
                ffn.extend(
                    [
                        self.activation_func,  # Activate
                        Dropout(
                            p=self.dropout_rate, inplace=False
                        ),  # Drop a fraction of the neurons out by setting them to zero during training.
                        Linear(
                            self.ffn_dim, self.ffn_dim
                        ),  # Add a layer to perform a linear transformation of the data
                    ]
                )
                if self.add_batch_norm:
                    ffn.append(BatchNorm1d(self.ffn_dim))

            ffn.extend(
                [
                    self.activation_func,  # Activate
                    Dropout(p=self.dropout_rate, inplace=False),
                ]
            )
            # print('ffn', ffn)
            self.ffn = Sequential(*ffn)
            self.final = Linear(
                self.ffn_dim, self.out_neurons, bias=True
            )  # bias specivies whether to include a bias term (Output = Input X Weight + Bias) or no. Adding this offset provides more flexibility for fitting the data

    def forward(self, x, edge_index, batch, global_feats=None):
        # print("data", data)
        x = x.float()
        # print(f"data.__dict__: {data.__dict__}")
        # print()
        # global_feats = data.global_feats if 'global_feats' in data.to_dict() else None
        # print(x.shape)

        # x_input = None
        # if add_skip_connection:
        #     ## Input for the skip connection
        #     x_input = x.clone()

        # print('global_feats is None ', global_feats is None)
        x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        # print(f'after conv1: {x is None}')
        for gconvo in self.convs:
            x = gconvo(x, edge_index)
            x = self.activation_func(x)
            # x = gconvo(x, edge_index)
            # x = Fun.tanh(x)
        # print(f'after convs: {x is None}')
        # print('x shape after convolutions: ', x.shape)

        if global_feats is None:
            ## This is applicable if we do not have global (molecule) features or they have already been added to
            ## each node of the respective molecules
            # print("POOLING", self.global_pooling(x,batch).shape)
            x = self.ffn(self.global_pooling(x, batch))
        else:
            ## This is applicable if the global (molecule) features have been computed, but not been added
            ## to the node features. Here, the node features are transformed to a vector, which is then
            ## condatenated with the global features
            # print('global feats', global_feats.shape) #, global_feats
            # print(self.global_pooling(x,batch).shape, global_feats.shape)
            # print(self.global_pooling(x,batch).dtype, global_feats.dtype)
            all_feats = cat(
                [self.global_pooling(x, batch), global_feats], dim=1
            ).float()
            # print(all_feats.shape, all_feats.dtype)
            # print(self.ffn.bias.shape, self.ffn.weight.shape)

            x = self.ffn(all_feats)

        # if add_skip_connection:
        #     # Add the skip connection
        #     x = x + x_input

        # print(f"x dim before final layer: {x.shape}")
        if self.ffn_nlayers > 1:
            x = self.final(x)
        # print("x = ", x.squeeze(1))
        if self.task in ["binary_classification", "multiclass_classification"]:
            # print("Before Softmax", x[:5])
            x = self.classification_out_layer(x)

        return x


class GAT(nn.Module):
    def __init__(
        self,
        task: str,
        in_channels,
        gnn_hidden_neurons: int = 128,
        gnn_nlayers: int = 2,
        global_fdim: int = None,
        edge_dim: int = None,
        heads: int = 8,
        ffn_hidden_neurons: int = 128,
        ffn_nlayers: int = 1,
        out_neurons: int = 1,
        dropout_rate: float = 0.3, ## feature dropout rate
        attn_dropout_rate: float = 0.3, ## attention coefficient dropout rate
        gpooling_func: str = "mean",
        activation_func=ReLU(),
        add_batch_norm: bool = False,
        add_edge_features: bool = False,
    ):
        super(GAT, self).__init__()

        assert (
            task.lower() in _allowable_tasks
        ), "The task must be either of the following: {'; '.join(_allowable_tasks)}."
        assert not (
            add_edge_features and edge_dim is None
        ), "Adding edge features requires to set edge_dim to a non-null value. However, edge_dim is None."

        self.task = task.lower()
        self.in_channels = in_channels
        self.gnn_dim = gnn_hidden_neurons
        self.gnn_nlayers = gnn_nlayers
        self.global_fdim = global_fdim
        self.edge_dim = edge_dim
        self.heads = heads
        self.ffn_dim = ffn_hidden_neurons
        self.ffn_nlayers = ffn_nlayers
        self.out_neurons = out_neurons
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.activation_func = activation_func
        self.add_edge_features = add_edge_features
        if isinstance(self.activation_func, str):
            self.activation_func = eval(f"{self.activation_func}()")

        self.add_batch_norm = add_batch_norm

        if gpooling_func == "mean":
            self.global_pooling = global_mean_pool
        elif gpooling_func == "sum":
            self.global_pooling = global_add_pool
        elif gpooling_func == "max":
            self.global_pooling = global_max_pool

        self.create_layers()
        self.initialize_params()
        # print(f"self.conv1 = \n{self.conv1}")

    def initialize_params(self):
        print("Initializing parameters...")
        for layer_ in self.modules():
            # print(layer_.__class__)
            if isinstance(layer_, GCNConv):
                layer_.reset_parameters()
            if isinstance(layer_, Linear):
                # print(f"\t{layer_.weight.shape}")
                init.xavier_uniform_(layer_.weight)
                init.zeros_(layer_.bias)

    @classmethod
    def from_dict(cls, params: dict):
        task = params.get("task").lower()
        in_channels = params.get("in_channels")
        gnn_hidden_neurons = params.get("gnn_hidden_neurons", 128)
        gnn_nlayers = params.get("gnn_nlayers", 2)
        global_fdim = params.get("global_fdim", None)
        edge_dim = params.get("edge_dim", None)
        heads = params.get("heads", 4)
        ffn_hidden_neurons = params.get("ffn_hidden_neurons", 128)
        ffn_nlayers = params.get("ffn_nlayers", 1)
        out_neurons = params.get("out_neurons", 1)
        dropout_rate = params.get("dropout_rate", 0.3)
        attn_dropout_rate = params.get("attn_dropout_rate", 0.3)
        activation_func = params.get("activation_func", ReLU())
        add_batch_norm = params.get("add_batch_norm", False)
        add_edge_features = params.get("add_edge_features", False)

        gpooling_func = params.get("gpooling_func", "mean")

        return cls(
            task=task,
            in_channels=in_channels,
            gnn_hidden_neurons=gnn_hidden_neurons,
            global_fdim=global_fdim,
            edge_dim=edge_dim,
            heads=heads,
            gnn_nlayers=gnn_nlayers,
            ffn_hidden_neurons=ffn_hidden_neurons,
            ffn_nlayers=ffn_nlayers,
            out_neurons=out_neurons,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            activation_func=activation_func,
            gpooling_func=gpooling_func,
            add_batch_norm=add_batch_norm,
            add_edge_features=add_edge_features,
        )

    def set_global_pooling_func(self, gpooling_func):
        if gpooling_func == "mean":
            self.global_pooling = global_mean_pool
        elif gpooling_func == "sum":
            self.global_pooling = global_add_pool
        elif gpooling_func == "max":
            self.global_pooling = global_max_pool


    # def create_conv_layer_sequence(self, dim_in, dim_out, heads, dropout_rate, seq_len=1):

    #     if self.add_edge_features:
    #         return ModuleList(
    #                             [GATv2Conv(
    #                                         dim_in,
    #                                         dim_out,
    #                                         edge_dim=self.edge_dim,
    #                                         heads=heads,
    #                                         dropout=attn_dropout_rate
    #                                     )

    #                             for x in range(seq_len)
    #                             ])
                                
    #     else:
    #         return ModuleList(
    #                             [GATv2Conv(
    #                                         dim_in,
    #                                         dim_out,
    #                                         heads=heads,
    #                                         dropout=attn_dropout_rate
    #                                     )

    #                             for x in range(seq_len)
    #                             ])            


    def create_layers(self):
        if self.task in ["binary_classification"]:
            self.classification_out_layer = Sigmoid()
        elif self.task in ["multiclass_classification"]:
            self.classification_out_layer = Softmax(dim=1)

        if self.add_edge_features:
            self.conv1 = GATv2Conv(
                in_channels = self.in_channels,
                out_channels = self.gnn_dim,
                edge_dim=self.edge_dim,
                heads=self.heads,
                dropout=self.attn_dropout_rate,
                # concat = True,
                # negative_slope=0.2
            )
        else:
            self.conv1 = GATv2Conv(
                in_channels = self.in_channels,
                out_channels = self.gnn_dim,
                heads=self.heads,
                dropout=self.attn_dropout_rate,
                # concat = True,
                # negative_slope=0.2
            )

        # If there are more than one graph convolutional layers.
        self.convs = ModuleList([])

        if self.add_edge_features:
            self.convs = ModuleList(
                [
                    GATv2Conv(
                        in_channels = self.gnn_dim * self.heads,
                        out_channels = self.gnn_dim,
                        edge_dim=self.edge_dim,
                        heads=self.heads,
                        dropout=self.attn_dropout_rate,
                        # concat = True,
                        # negative_slope=0.2
                    )
                    for x in range(self.gnn_nlayers - 1)
                ]

            )
        else:
            self.convs = ModuleList(
                [
                    GATv2Conv(
                        in_channels = self.gnn_dim * self.heads,
                        out_channels = self.gnn_dim,
                        heads=self.heads,
                        dropout=self.attn_dropout_rate,
                        # concat = True,
                        # negative_slope=0.2
                    )
                    for x in range(self.gnn_nlayers - 1)
                ]

            )       
        
        # self.conv1 =  self.create_conv_layer_sequence(seq_len=1, dim_in=self.in_channels, dim_out=self.gnn_dim, heads=self.heads, dropout_rate=self.dropout_rate)[0]
        # print(f"self.conv1 = {self.conv1}")

        # if self.gnn_nlayers>=2:
        #     self.convs = self.create_conv_layer_sequence(dim_in=self.in_channels*self.heads, dim_out=self.gnn_dim, heads=self.heads, dropout_rate=self.dropout_rate, seq_len=self.gnn_nlayers-1)




        ## Build the Feed-forward neural network

        if self.ffn_nlayers == 1:
            if not self.global_fdim is None:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(
                        (self.gnn_dim * self.heads) + self.global_fdim, self.out_neurons
                    )
                    # Linear((self.gnn_dim) + self.global_fdim, self.out_neurons)
                ]
            else:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim * self.heads, self.out_neurons)
                ]

            if self.add_batch_norm:
                ffn.append(BatchNorm1d(self.out_neurons))

            self.ffn = Sequential(*ffn)

        elif self.ffn_nlayers > 1:
            # ffn = [
            #         Dropout(p=self.dropout_rate, inplace=False),
            #         Linear(self.gnn_dim * self.heads, self.ffn_dim)
            #     ]

            if not self.global_fdim is None:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear((self.gnn_dim * self.heads) + self.global_fdim, self.ffn_dim)
                    # Linear((self.gnn_dim) + self.global_fdim, self.ffn_dim)
                ]
            else:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim * self.heads, self.ffn_dim)
                    # Linear(self.gnn_dim, self.ffn_dim)
                ]

            if self.add_batch_norm:
                ffn.append(BatchNorm1d(self.ffn_dim))

            for x in range(self.ffn_nlayers - 2):
                ffn.extend(
                    [
                        self.activation_func,  # Activate
                        Dropout(
                            p=self.dropout_rate, inplace=False
                        ),  # Drop a fraction of the neurons out by setting them to zero during training.
                        Linear(
                            self.ffn_dim, self.ffn_dim
                        ),  # Add a layer to perform a linear transformation of the data
                    ]
                )

            ffn.extend(
                [
                    self.activation_func,  # Activate
                    Dropout(p=self.dropout_rate, inplace=False),
                ]
            )

            self.ffn = Sequential(*ffn)
            self.final = Linear(
                self.ffn_dim, self.out_neurons, bias=True
            )  # bias specivies whether to include a bias term (Output = Input X Weight + Bias) or no. Adding this offset provides more flexibility for fitting the data

    def forward(self, x, edge_index, batch, global_feats=None, edge_attr=None):
        # def forward(self, data):
        # print("data", data)
        # x, edge_index, edge_attr, dbatch = data.x.float(), data.edge_index, data.edge_attr, data.batch
        # global_feats = data.global_feats if 'global_feats' in data.to_dict() else None
        
        # print("x size = ", x.shape)
        # x = Dropout(p=self.dropout_rate, inplace=False)(x)
        if self.add_edge_features and not edge_attr is None:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        x = Dropout(p=self.dropout_rate, inplace=False)(x)

        for i in range(len(self.convs)):
            gconvo = self.convs[i]
            # x = Dropout( p=self.dropout_rate, inplace=False)(x)
            if self.add_edge_features and not edge_attr is None:
                x = gconvo(x, edge_index, edge_attr=edge_attr)
            else:
                x = gconvo(x, edge_index)
            x = self.activation_func(x)

            if i < len(self.convs) - 1:
                x = Dropout(p=self.dropout_rate, inplace=False)(x)

        # print('x shape after convolutions: ', x.shape)

        if global_feats is None:
            ## This is applicable if we do not have global (molecule) features or they have already been added to
            ## each node of the respective molecules
            # print(self.global_pooling(x,dbatch).shape)
            x = self.ffn(global_mean_pool(x, batch))
            # print('After global_mean_pool', x.shape)
        else:
            ## This is applicable if the global (molecule) features have been computed, but not been added
            ## to the node features. Here, the node features are transformed to a vector, which is then
            ## condatenated with the global features
            # print(self.global_pooling(x,dbatch).shape, global_feats.shape)
            # print(self.global_pooling(x,dbatch).dtype, global_feats.dtype)
            # print(self.global_pooling(x,dbatch).to(dtype=torch.float64).dtype)
            # print(global_feats.to(dtype=torch.float64).dtype)

            # x = self.ffn(cat([self.global_pooling(x,dbatch).to(dtype=torch.float64), global_feats.to(dtype=torch.float64)], dim=1)) #.float()
            all_feats = cat(
                [self.global_pooling(x, batch), global_feats], dim=1
            ).float()
            # print('all_feats ', all_feats.shape)
            # time.sleep(5)

            x = self.ffn(all_feats)

        # print(f"x dim before final layer: {x.shape}")
        if self.ffn_nlayers > 1:
            x = self.final(x)

        if self.task in ["binary_classification", "multiclass_classification"]:
            # print("Before Softmax", x[:5])
            x = self.classification_out_layer(x)
            # print(x)

        return x

    def get_embedding(self, data):
        # print("data", data)
        x, edge_index, edge_attr, dbatch = (
            data.x.float(),
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        global_feats = data.global_feats if "global_feats" in data.to_dict() else None

        # x = Dropout(p=self.dropout_rate, inplace=False)(x)
        if self.add_edge_features:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        x = Dropout(p=self.dropout_rate, inplace=False)(x)

        for i in range(len(self.convs)):
            gconvo = self.convs[i]
            x = Dropout(p=self.dropout_rate, inplace=False)(x)
            if self.add_edge_features:
                x = gconvo(x, edge_index, edge_attr=edge_attr)
            else:
                x = gconvo(x, edge_index)
            x = self.activation_func(x)

            if i < len(self.convs) - 1:
                x = Dropout(p=self.dropout_rate, inplace=False)(x)

        if global_feats is None:
            x = self.global_pooling(x, dbatch).float()
        else:
            x = cat([self.global_pooling(x, dbatch), global_feats], dim=1).float()       

        return x


class GIN(nn.Module):
    def __init__(
        self,
        task: str,
        in_channels: int,
        global_fdim: int = None,
        gnn_hidden_neurons: int = 128,
        gnn_nlayers: int = 2,
        ffn_hidden_neurons: int = 128,
        ffn_nlayers: int = 1,
        out_neurons: int = 1,
        dropout_rate: float = 0.3,
        gpooling_func: str = "sum",
        activation_func: [str, object] = torch.nn.LeakyReLU(),
    ):
        super(GIN, self).__init__()

        assert (
            task.lower() in _allowable_tasks
        ), "The task must be either of the following: {'; '.join(_allowable_tasks)}."
        assert (
            gpooling_func.lower() in _allowable_pooling_funcs
        ), "The global poling function must be either of the following: {'; '.join(_allowable_pooling_funcs)}."

        self.task = task.lower()
        self.in_channels = in_channels
        self.gnn_dim = gnn_hidden_neurons
        self.global_fdim = global_fdim
        self.gnn_nlayers = gnn_nlayers
        self.ffn_dim = ffn_hidden_neurons
        self.ffn_nlayers = ffn_nlayers
        self.out_neurons = out_neurons
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        if isinstance(self.activation_func, str):
            self.activation_func = eval(f"{self.activation_func}()")

        self.set_global_pooling_func(gpooling_func)
        self.create_layers()
        self.initialize_params

    def initialize_params(self):
        print("Initializing parameters...")
        for layer_ in self.modules():
            # print(layer_.__class__)
            if isinstance(layer_, GCNConv):
                layer_.reset_parameters()
            if isinstance(layer_, Linear):
                # print(f"\t{layer_.weight.shape}")
                init.xavier_uniform_(layer_.weight)
                init.zeros_(layer_.bias)

    @classmethod
    def from_dict(cls, params: dict):
        task = params.get("task").lower()
        in_channels = params.get("in_channels")
        gnn_hidden_neurons = params.get("gnn_hidden_neurons", 128)
        global_fdim = params.get("global_fdim", None)
        gnn_nlayers = params.get("gnn_nlayers", 2)
        ffn_hidden_neurons = params.get("ffn_hidden_neurons", 128)
        ffn_nlayers = params.get("ffn_nlayers", 1)
        out_neurons = params.get("out_neurons", 1)
        dropout_rate = params.get("dropout_rate", 0.3)
        activation_func = params.get("activation_func", torch.nn.ReLU())
        gpooling_func = params.get("gpooling_func", "sum")

        return cls(
            task=task,
            in_channels=in_channels,
            gnn_hidden_neurons=gnn_hidden_neurons,
            global_fdim=global_fdim,
            gnn_nlayers=gnn_nlayers,
            ffn_hidden_neurons=ffn_hidden_neurons,
            ffn_nlayers=ffn_nlayers,
            out_neurons=out_neurons,
            dropout_rate=dropout_rate,
            activation_func=activation_func,
            gpooling_func=gpooling_func,
        )

    def set_global_pooling_func(self, gpooling_func):
        if gpooling_func == "mean":
            self.global_pooling = global_mean_pool
        elif gpooling_func == "sum":
            self.global_pooling = global_add_pool
        elif gpooling_func == "max":
            self.global_pooling = global_max_pool

    def create_layers(self):
        if self.task in ["binary_classification"]:
            self.classification_out_layer = Sigmoid()
        elif self.task in ["multiclass_classification"]:
            self.classification_out_layer = Softmax(dim=1)

        print(
            f"in_channels = {self.in_channels} :: gnn_dim = {self.gnn_dim} :: activation_func = {self.activation_func}"
        )
        self.conv1 = GINConv(
            Sequential(
                Linear(self.in_channels, self.gnn_dim),
                BatchNorm1d(self.gnn_dim),
                self.activation_func,
                Linear(self.gnn_dim, self.gnn_dim),
                self.activation_func,
            )
        )

        ## If there are more than one graph convolutional layers.
        self.convs = ModuleList(
            [
                GINConv(
                    Sequential(
                        Linear(self.gnn_dim, self.gnn_dim),
                        BatchNorm1d(self.gnn_dim),
                        self.activation_func,
                        Linear(self.gnn_dim, self.gnn_dim),
                        self.activation_func,
                    )
                )
                for x in range(self.gnn_nlayers - 1)
            ]
        )

        ## Build the Feed-forward neural network

        if self.ffn_nlayers == 1:
            if self.global_fdim is None:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim, self.out_neurons),
                ]
            else:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear((self.gnn_dim) + self.global_fdim, self.out_neurons),
                ]
            self.ffn = Sequential(*ffn)

        elif self.ffn_nlayers > 1:
            if self.global_fdim is None:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear(self.gnn_dim, self.ffn_dim),
                ]
            else:
                ffn = [
                    Dropout(p=self.dropout_rate, inplace=False),
                    Linear((self.gnn_dim) + self.global_fdim, self.ffn_dim),
                ]

            for x in range(self.ffn_nlayers - 2):
                ffn.extend(
                    [
                        self.activation_func,  # Activate
                        Dropout(
                            p=self.dropout_rate, inplace=False
                        ),  # Drop a fraction of the neurons out by setting them to zero during training.
                        Linear(
                            self.ffn_dim, self.ffn_dim
                        ),  # Add a layer to perform a linear transformation of the data
                    ]
                )

            ffn.extend(
                [
                    self.activation_func,  # Activate
                    Dropout(p=self.dropout_rate, inplace=False),
                ]
            )

            self.ffn = Sequential(*ffn)
            self.final = Linear(
                self.ffn_dim, self.out_neurons, bias=True
            )  # bias specivies whether to include a bias term (Output = Input X Weight + Bias) or no. Adding this offset provides more flexibility for fitting the data

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for f in self.ffn:
            if f.__class__.__name__ == "Linear":
                f.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, edge_index, batch, global_feats=None):
        # def forward(self, data):
        # print("data", data)

        # x, edge_index, dbatch = data.x, data.edge_index, data.batch
        # global_feats = data.global_feats if 'global_feats' in data.to_dict() else None

        ## Node embeddings
        x = self.conv1(x, edge_index)
        # print(f"x[0] = {x[0].shape}")

        for gconvo in self.convs:
            x = gconvo(x, edge_index)

        x = self.global_pooling(x, batch)

        if not global_feats is None:
            # print('global_feats', global_feats.shape)
            x = cat((x, global_feats), dim=1).to(dtype=torch.float32)

        # print('x', x.shape, x.dtype)
        # print(self.ffn.state_dict().keys())
        # print(self.ffn.state_dict()['1.weight'].shape, self.ffn.state_dict()['1.bias'].shape)
        x = self.ffn(x)
        # print('x2', x.shape)

        # print(f"x dim before final layer: {x.shape}")
        if self.ffn_nlayers > 1:
            x = self.final(x)

        if self.task in ["binary_classification", "multiclass_classification"]:
            # print("Before Softmax", x[:5])
            x = self.classification_out_layer(x)

        # print(f"H = {h}")
        return x
