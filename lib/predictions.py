from typing import List, Union, Any
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import cat, tensor
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import inspect

from lib import datasets, gnn_utils, graph_nns, datasets, featurizers as feat
from lib.featurizers import (
    AtomFeaturizer,
    BondFeaturizer,
    MoleculeFeaturizer,
    get_node_dict_from_atom_featurizer,
)


def predict_from_batch(batch, model, return_true_targets: bool = False):
    output = None
    if type(model) in [graph_nns.GCN, graph_nns.GAT, graph_nns.GIN]:
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


def predict_from_loader(
    model, loader, device="cpu", return_true_targets: bool = False, desc="Predicting..."
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


def predict_from_data_list(
    data: Union[Data, List[Data]],
    batch_size=128,
    device="cpu",
    shuffle: bool = False,
    add_global_feats_to_nodes: bool = False,
    num_workers: int = 0,
    scale_features: bool = True,
    feature_scaler=MinMaxScaler(),
    desc="Predicting...",
):
    loader = None
    if isinstance(data, List):
        loader = datasets.get_dataloader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            num_workers=num_workers,
            scale_features=scale_features,
            feature_scaler=feature_scaler,
        )

    elif isinstance(data, Data):
        loader = datasets.get_dataloader(
            dataset=[data],
            batch_size=batch_size,
            shuffle=shuffle,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            num_workers=num_workers,
            scale_features=scale_features,
            feature_scaler=feature_scaler,
        )
    else:
        raise TypeError(
            "data must be an instance of the following classes: Data, or List[Data]."
        )

    return self.predict_from_loader(loader, device=device, desc=desc)


# def predict_from_data_list(self, data:Union[Data, List[Data]], batch_size=128, device='cpu'
#                         , desc="Predicting..."):
#     loader = None
#     if isinstance(data, List):
#         loader = DataLoader(dataset=data, batch_size=batch_size)
#     elif isinstance(data, Data):
#         loader = DataLoader(dataset=[data], batch_size=batch_size)
#     else:
#         raise TypeError("data must be an instance of the following classes: Data, or List.")

#     return self.predict_from_loader(loader, device=device, desc=desc)


def predict_from_smiles_list(
    smiles_list: List[str],
    batch_size=128,
    device="cpu",
    shuffle=False,
    num_workers: int = 0,
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: BondFeaturizer = feat.BOND_FEATURIZER,
    mol_featurizer: feat.MoleculeFeaturizer = feat.MoleculeFeaturizer(),
    add_explicit_h: bool = True,
    compute_global_features: bool = True,
    add_global_feat_to_nodes: bool = False,
    scale_features: bool = True,
    feature_scaler=MinMaxScaler(),
    desc="Predicting...",
):
    graphs = gnn_utils.graph_from_smiles_list(
        smiles_list,
        add_explicit_h=add_explicit_h,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
        compute_global_features=compute_global_features,
        add_global_feat_to_nodes=add_global_feat_to_nodes,
    )

    return self.predict_from_data_list(
        graphs,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
        scale_features=scale_features,
        feature_scaler=feature_scaler,
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
        batch_size=128,
        device="cpu",
        shuffle=False,
        num_workers: int = 0,
        return_true_targets: bool = False,
        desc="Predicting...",
    ):
        loader = None
        if isinstance(data, List):
            loader = datasets.get_dataloader(
                dataset=data,
                batch_size=batch_size,
                shuffle=False,
                add_global_feats_to_nodes=self.add_global_feats_to_nodes,
                num_workers=num_workers,
                scale_features=self.scale_features,
                feature_scaler=self.feature_scaler,
            )

        elif isinstance(data, Data):
            loader = datasets.get_dataloader(
                dataset=[data],
                batch_size=batch_size,
                shuffle=False,
                add_global_feats_to_nodes=self.add_global_feats_to_nodes,
                num_workers=num_workers,
                scale_features=self.scale_features,
                feature_scaler=self.feature_scaler,
            )
        else:
            raise TypeError(
                "data must be an instance of the following classes: Data, or List[Data]."
            )

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
