from typing import List, Union, Any
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import cat, tensor
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import inspect
import warnings

from lib import datasets, gnn_utils, graph_nns, datasets, featurizers as feat
from lib.featurizers import (
    AtomFeaturizer,
    BondFeaturizer,
    MoleculeFeaturizer,
    get_node_dict_from_atom_featurizer,
)


def predict_from_batch(batch:Batch, model:graph_nns.MyGNN, return_true_targets: bool = False):
    
    output = None
    assert isinstance(model,graph_nns.MyGNN), f"ValueError: The current model of class {model.__class__} is not a subclass of graph_nns.MyGNN."

    batch.x = batch.x.float()
    # output = model(batch)
    global_feats = batch.to_dict().get("global_feats", None)

    if not 'edge_attr' in inspect.signature(model.forward).parameters:
        output = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, global_feats=global_feats)
    else:
        edge_attr_ = batch.to_dict().get('edge_attr', None)
        # print(f"edge_attr = {edge_attr_.shape or None}")
        output = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch
                        , global_feats=global_feats, edge_attr=batch.to_dict().get("edge_attr", None))


    if return_true_targets:
        # print(batch.y)
        return output, batch.y.float()
    else:
        return output, None


def predict_from_loader(model:graph_nns.MyGNN, loader:DataLoader, device: str = "cpu"
                        , return_true_targets: bool = False, desc:str = "Predicting..."
):
    pred_target = torch.empty((0,), dtype=torch.float32).to(device)
    # outputs = torch.empty((0,model.out_neurons), dtype=torch.float32).to(device)
    true_target = torch.empty((0,), dtype=torch.float32).to(device)
    exportable_y = False
    if return_true_targets and hasattr(list(loader)[0], "y"):
        exportable_y = True

    task = model.task
    model = model.to(device)

    for data in loader:
        # for data in tqdm(loader, desc=desc): #
        data = data.to(device)

        output, true_y = predict_from_batch(
            batch=data, model=model, return_true_targets=return_true_targets
        )

        if task in ["binary_classification"]:
            pred_target = cat((pred_target, output), dim=0)
        elif task == "multilabel_classification":
            _, predicted = torch.max(
                output, 1
            )  ## returns maximum values(_), and their indices (predicted) along the dimmension 1
            pred_target = cat((pred_target, predicted), dim=0)

        elif task == "regression":
            pred_target = cat((pred_target, output.float()), dim=0)
        else:
            raise ValueError(f"No implementation for task {task}.")
        # outputs = cat((outputs, output), dim=0)

        if exportable_y:
            true_target = cat((true_target, true_y.float()), dim=0)

    if return_true_targets:
        return pred_target, true_target
    else:
        return pred_target


def predict_from_loader(
    model:graph_nns.MyGNN,
    loader:DataLoader,
    device: str="cpu",
    return_true_targets: bool = False,
    desc:str ="Predicting...",
):
    pred_target = torch.empty((0,), dtype=torch.float32).to(device)
    # outputs = torch.empty((0,model.out_neurons), dtype=torch.float32).to(device)
    true_target = torch.empty((0,), dtype=torch.float32).to(device)
    exportable_y = False

    # print("device = ", device)
    # print(loader)
    first_batch = None
    for b in loader:
        # print("b= ", b)
        # print(dir(b))
        first_batch = b.to(device)
        break
    # first_batch = next(iter(loader)) #.to(device)
    # print(first_batch)

    if return_true_targets and hasattr(first_batch, "y"):
        exportable_y = True
        # print("exportable_y = True")

    task = model.task
    model = model.to(device)

    for data in loader:
        # for data in tqdm(loader, desc=desc): #
        # data   = data.to(device)

        output, true_y = predict_from_batch(
            batch=data.to(device), model = model, return_true_targets=return_true_targets
        )

        if task in ["binary_classification"]:
            pred_target = cat((pred_target, output), dim=0)
        elif task == "multilabel_classification":
            _, predicted = torch.max(
                output, 1
            )  ## returns maximum values(_), and their indices (predicted) along the dimmension 1
            pred_target = cat((pred_target, predicted), dim=0)

        elif task == "regression":
            pred_target = cat((pred_target, output.float()), dim=0)
        else:
            raise ValueError(f"No implementation for task {task}.")
        # outputs = cat((outputs, output), dim=0)

        if exportable_y:
            true_target = cat((true_target, true_y.float()), dim=0)

    if return_true_targets:
        return pred_target, true_target
    else:
        return pred_target

def predict_from_data_list(
    model:graph_nns.MyGNN,
    data: Union[Data, List[Data]],
    add_global_feats_to_nodes:bool = False,
    batch_size: int=128,
    device: str="cpu",
    shuffle: bool=False,
    num_workers: int = 0,
    return_true_targets: bool = False,

    desc: str="Predicting...",
):

    print("add_global_feats_to_nodes", add_global_feats_to_nodes)
    if add_global_feats_to_nodes and (not hasattr(data[0], 'global_feats')):
        x_dim = model.in_channels
        print(f"data[0].x.shape[1] = {data[0].x.shape[1]}  / x_dim = {x_dim}", )
        if data[0].x.shape[1] == x_dim:
            warnings.warn("\n\n*******************************At least one data item does not have a global_feats. However, the x parameter has a size of {data[0].x.shape[1]}, which is equal to the expected value of {x_dim}. So we suppose global features have been added already.\nThe global feature addition step will be skipped.\n*******************************")
            add_global_feats_to_nodes = False
        else:
            raise Exception(f"\n\n*******************************There are no features to add. The current x attribute of size {data[0].x.shape[1]} indicates that no features were added to the original {x_dim} node features features.")
    
    loader = None
    if isinstance(data, List):
        loader = datasets.get_dataloader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            # add_global_feats_to_nodes=False,
            num_workers=num_workers,
        )

    elif isinstance(data, Data):
        loader = datasets.get_dataloader(
            dataset=[data],
            batch_size=batch_size,
            shuffle=False,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            # add_global_feats_to_nodes=False,
            num_workers=num_workers,
        )
    else:
        raise TypeError(
            "data must be an instance of the following classes: Data, or List[Data]."
        )

    print("loader: ", next(iter(loader)))

    # print("loader", list(loader))
    return predict_from_loader(model=model,
        loader=loader, device=device, return_true_targets=return_true_targets, desc=desc
    )


def predict_from_smiles_list(
    model:graph_nns.MyGNN,
    smiles_list: List[str],
    batch_size:int =128,
    device: str="cpu",
    add_explicit_h: bool=False,
    atom_featurizer:AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer:AtomFeaturizer = feat.BOND_FEATURIZER,
    mol_featurizer:AtomFeaturizer  = feat.MoleculeFeaturizer(),
    add_global_feats_to_nodes:bool = False,
    shuffle: bool=False,
    num_workers: int = 0,
    desc: str="Predicting...",
):
    graphs = gnn_utils.graph_from_smiles_list(
        smiles_list,
        add_explicit_h=add_explicit_h,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
    )

    return predict_from_data_list(
        model=model,
        data=graphs,
        add_global_feats_to_nodes = add_global_feats_to_nodes,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
        num_workers=num_workers,
        desc=desc,
    )





class GNNPredictor:
    def __init__(
        self,
        model,
        atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
        bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
        add_explicit_h: bool = True,
        mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
        scale_features: bool = True,
        feature_scaler=StandardScaler(),
        compute_global_features: bool = True,
        add_global_feats_to_nodes: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.mol_featurizer = mol_featurizer
        self.scale_features = scale_features
        self.feature_scaler = feature_scaler
        self.add_explicit_h = add_explicit_h
        self.compute_global_features = compute_global_features
        self.add_global_feats_to_nodes = add_global_feats_to_nodes
        # print(dir(self.atom_featurizer))
        # print(self.atom_featurizer.features_mapping)
        self.node_dict = get_node_dict_from_atom_featurizer(self.atom_featurizer)

    def set_model(self, new_model):
        self.model = new_model
        self.model.eval()

    def predict_from_batch(self, batch, return_true_targets: bool = False):
        output = None
        if type(self.model) in [graph_nns.GCN, graph_nns.GAT, graph_nns.GIN]:
            batch.x = batch.x.float()
            # output = model(batch)
            global_feats = batch.to_dict().get("global_feats", None)
            output = self.model(batch.x, batch.edge_index, batch.batch, global_feats)

        if return_true_targets:
            # print(batch.y)
            return output, batch.y.float()
        else:
            return output, None

    def predict_from_loader(
        self,
        loader,
        device="cpu",
        return_true_targets: bool = False,
        desc="Predicting...",
    ):
        pred_target = torch.empty((0,), dtype=torch.float32).to(device)
        # outputs = torch.empty((0,model.out_neurons), dtype=torch.float32).to(device)
        true_target = torch.empty((0,), dtype=torch.float32).to(device)
        exportable_y = False

        # print("device = ", device)
        # print(loader)
        first_batch = None
        for b in loader:
            # print("b= ", b)
            # print(dir(b))
            first_batch = b.to(device)
            break
        # first_batch = next(iter(loader)) #.to(device)
        # print(first_batch)

        if return_true_targets and hasattr(first_batch, "y"):
            exportable_y = True
            # print("exportable_y = True")

        task = self.model.task
        model = self.model.to(device)

        for data in loader:
            # for data in tqdm(loader, desc=desc): #
            # data   = data.to(device)

            output, true_y = self.predict_from_batch(
                batch=data.to(device), return_true_targets=return_true_targets
            )

            if task in ["binary_classification"]:
                pred_target = cat((pred_target, output), dim=0)
            elif task == "multilabel_classification":
                _, predicted = torch.max(
                    output, 1
                )  ## returns maximum values(_), and their indices (predicted) along the dimmension 1
                pred_target = cat((pred_target, predicted), dim=0)

            elif task == "regression":
                pred_target = cat((pred_target, output.float()), dim=0)
            else:
                raise ValueError(f"No implementation for task {task}.")
            # outputs = cat((outputs, output), dim=0)

            if exportable_y:
                true_target = cat((true_target, true_y.float()), dim=0)

        if return_true_targets:
            return pred_target, true_target
        else:
            return pred_target

    def predict_from_data_list(
        self,
        data: Union[Data, List[Data]],
        add_global_feats_to_nodes:bool = None,
        batch_size=128,
        device="cpu",
        shuffle=False,
        num_workers: int = 0,
        return_true_targets: bool = False,

        desc="Predicting...",
    ):
        if add_global_feats_to_nodes is None:
            add_global_feats_to_nodes=self.add_global_feats_to_nodes

        print("add_global_feats_to_nodes", add_global_feats_to_nodes)
        
        has_global_feats      = True
        current_num_feats_x   = None 
        x_dim = self.model.in_channels

        if isinstance(data, List):
            has_global_feats = hasattr(data[0], 'global_feats')
            current_num_feats_x = data[0].x.shape[1]


        else:
            has_global_feats = hasattr(data, 'global_feats')
            current_num_feats_x = data.x.shape[1]

        # print("current_x_size = " {current_x_size})

        if add_global_feats_to_nodes and not has_global_feats:
            
            print(f"current_num_feats_x = {current_num_feats_x}  / x_dim = {x_dim}", )
            if current_num_feats_x == x_dim:
                warnings.warn("\n\n*******************************At least one data item does not have a global_feats. However, the x parameter has a size of {current_num_feats_x}, which is equal to the expected value of {x_dim}. So we suppose global features have been added already.\nThe global feature addition step will be skipped.\n*******************************")
                add_global_feats_to_nodes = False
            elif current_num_feats_x < x_dim:
                raise Exception(f"\n\n*******************************There are no features to add. The current x attribute of size {current_num_feats_x} indicates that no features were added to the original {x_dim} node features features.")
            else:
                raise Exception(f"\n\n******************************* Current number of features ({current_num_feats_x}) exceeds the expected number ({x_dim}).")


        loader = None
        if isinstance(data, List):
            loader = datasets.get_dataloader(
                dataset=data,
                batch_size=batch_size,
                shuffle=False,
                add_global_feats_to_nodes=add_global_feats_to_nodes,
                # add_global_feats_to_nodes=False,
                num_workers=num_workers,
            )

        elif isinstance(data, Data):
            loader = datasets.get_dataloader(
                dataset=[data],
                batch_size=batch_size,
                shuffle=False,
                add_global_feats_to_nodes=add_global_feats_to_nodes,
                # add_global_feats_to_nodes=False,
                num_workers=num_workers,
            )
        else:
            raise TypeError(
                "data must be an instance of the following classes: Data, or List[Data]."
            )

        print("loader: ", next(iter(loader)))

        # print("loader", list(loader))
        return self.predict_from_loader(
            loader, device=device, return_true_targets=return_true_targets, desc=desc
        )

    def predict_from_smiles_list(
        self,
        smiles_list: List[str],
        batch_size=128,
        device="cpu",
        shuffle=False,
        num_workers: int = 0,
        desc="Predicting...",
    ):
        graphs = gnn_utils.graph_from_smiles_list(
            smiles_list,
            add_explicit_h=self.add_explicit_h,
            atom_featurizer=self.atom_featurizer,
            bond_featurizer=self.bond_featurizer,
            mol_featurizer=self.mol_featurizer,
        )

        return self.predict_from_data_list(
            graphs,
            batch_size=batch_size,
            device=device,
            shuffle=shuffle,
            num_workers=num_workers,
            desc=desc,
        )
