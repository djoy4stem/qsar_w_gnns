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

from rdkit.Chem import MolFromSmiles
from lib import datasets, gnn_utils, graph_nns, datasets, featurizers as feat
from lib.featurizers import (
    AtomFeaturizer,
    BondFeaturizer,
    MoleculeFeaturizer,
    get_node_dict_from_atom_featurizer,
)
from torch.nn import functional as F
from lib import visuals, utilities


ALLOWABLE_MULTI_CLASS_TYPES_ = ["logits", "probas", "labels"]


def predict_from_batch(
    batch: Batch, model: graph_nns.MyGNN, return_true_targets: bool = False
):
    output = None
    assert isinstance(
        model, graph_nns.MyGNN
    ), f"ValueError: The current model of class {model.__class__} is not a subclass of graph_nns.MyGNN."

    batch.x = batch.x.float()
    # output = model(batch)
    global_feats = batch.to_dict().get("global_feats", None)

    if not "edge_attr" in inspect.signature(model.forward).parameters:
        output = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            global_feats=global_feats,
        )
    else:
        edge_attr_ = batch.to_dict().get("edge_attr", None)
        # print(f"edge_attr = {edge_attr_.shape or None}")
        output = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            global_feats=global_feats,
            edge_attr=batch.to_dict().get("edge_attr", None),
        )

    if return_true_targets:
        # print("batch.y = ", batch.y)
        return output, batch.y.float()
    else:
        return output, None


def predict_from_loader(
    model: graph_nns.MyGNN,
    loader: DataLoader,
    multi_class_type: str = "logits",
    device: str = "cpu",
    return_true_targets: bool = False,
    return_class_as_proba: bool = False,
    desc: str = "Predicting...",
    **kwargs,
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
            batch=data.to(device), model=model, return_true_targets=return_true_targets
        )

        if task in ["binary_classification"]:
            output = F.sigmoid(output)
            # print("bin output", output)
            if not return_class_as_proba:
                threshold = kwargs.get("binary_threshold", 0.5)
                output = torch.as_tensor([int(x > threshold) for x in output]).to(
                    device
                )
            pred_target = cat((pred_target, output), dim=0)

        elif task == "multiclass_classification":
            output = F.softmax(output)
            if return_class_as_proba:
                output = F.softmax(output)
            else:
                # _, predicted = torch.max(
                #     output, 1
                # )  ## returns maximum values(_), and their indices (predicted) along the dimmension 1
                pred_target = cat((pred_target, output), dim=0)

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
    model: graph_nns.MyGNN,
    data: Union[Data, List[Data]],
    add_global_feats_to_nodes: bool = False,
    batch_size: int = 128,
    device: str = "cpu",
    shuffle: bool = False,
    num_workers: int = 0,
    return_class_as_proba: bool = False,
    return_true_targets: bool = False,
    desc: str = "Predicting...",
    **kwargs,
):
    print("add_global_feats_to_nodes", add_global_feats_to_nodes)
    if add_global_feats_to_nodes and (not hasattr(data[0], "global_feats")):
        x_dim = model.in_channels
        print(
            f"data[0].x.shape[1] = {data[0].x.shape[1]}  / x_dim = {x_dim}",
        )
        if data[0].x.shape[1] == x_dim:
            warnings.warn(
                "\n\n*******************************At least one data item does not have a global_feats. However, the x parameter has a size of {data[0].x.shape[1]}, which is equal to the expected value of {x_dim}. So we suppose global features have been added already.\nThe global feature addition step will be skipped.\n*******************************"
            )
            add_global_feats_to_nodes = False
        else:
            raise Exception(
                f"\n\n*******************************There are no features to add. The current x attribute of size {data[0].x.shape[1]} indicates that no features were added to the original {x_dim} node features features."
            )

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
    return predict_from_loader(
        model=model,
        loader=loader,
        device=device,
        return_class_as_proba=return_class_as_proba,
        return_true_targets=return_true_targets,
        desc=desc,
        **kwargs,
    )


def predict_from_smiles_list(
    model: graph_nns.MyGNN,
    smiles_list: List[str],
    return_class_as_proba: bool = False,
    batch_size: int = 128,
    device: str = "cpu",
    add_explicit_h: bool = False,
    atom_featurizer: AtomFeaturizer = feat.ATOM_FEATURIZER,
    bond_featurizer: AtomFeaturizer = feat.BOND_FEATURIZER,
    mol_featurizer: AtomFeaturizer = feat.MoleculeFeaturizer(),
    add_global_feats_to_nodes: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    desc: str = "Predicting...",
    **kwargs,
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
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
        num_workers=num_workers,
        return_class_as_proba=return_class_as_proba,
        desc=desc,
        **kwargs,
    )


def predict_and_compute_contributions(
    model: graph_nns.MyGNN,
    atom_featurizer: AtomFeaturizer,
    bond_featurizer: BondFeaturizer,
    mol_featurizer: MoleculeFeaturizer,
    smiles: str,
    device: str = "cpu",
    normalize_contributions: bool = True,
):
    model = model.to(device).eval()

    # Ensure model compatibility and move to device
    if model.global_fdim is None or model.global_fdim <= 0:
        raise ValueError(
            f"Global feature dimension (global_fdim) must be >0 for contributions. "
            f"Current model's global_fdim: {model.global_fdim}"
        )

    data = gnn_utils.graph_from_molecule(
        molecule=MolFromSmiles(smiles),
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
        add_explicit_h=False,
        compute_global_features=True,
        add_global_feat_to_nodes=False,
    ).to(device)

    # Enable gradient tracking on fingerprint for all operations performed on it. It helps
    # computing the gradients of the model's output w.r.t each feature

    data.global_feats.requires_grad = True
    ## Forward pass
    # We make a forward pass through the model to get the prediction score or pobability
    output = model(data.x, data.edge_index, data.batch, data.global_feats)
    predicted_score = output.squeeze()
    # print("predicted score:", predicted_score)

    ## Gradient calculation / gradient-based attribution
    # We Compute gradients of predicted score w.r.t to the input features. These gradients show
    # how sensitiv ethe output is to changes in each individual input feature
    # allowing us to interpret their importance
    # The resulting gradients act as a feature attribution map, indicating with parts of the input molecule
    # had the largest effect on pushing the model toward or away from the final prediction.
    # The prediction score need to be used further. It is just a starting point  for the gradient calculation
    predicted_score.backward()

    contributions = (
        data.global_feats.grad * data.global_feats
    )  # Element-wise product for contributions

    if normalize_contributions:
        contributions = utilities.min_max_normalize(contributions)
        # contributions = F.softmax(contributions, dim=0)

    return predicted_score, contributions.squeeze(0).detach()


def predict_and_visualize_subsructure_contributions(
    model: graph_nns.MyGNN,
    smiles: str,
    func_groups: Union[List, str],
    atom_featurizer: AtomFeaturizer,
    bond_featurizer: BondFeaturizer,
    mol_featurizer: MoleculeFeaturizer,
    threshold: float = 0.5,
    scale_type="log",
    device: str = "cpu",
    normalize_contributions: bool = True,
):
    prediction, contributions = predict_and_compute_contributions(
        model=model,
        smiles=smiles,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        mol_featurizer=mol_featurizer,
        normalize_contributions=normalize_contributions,
    )

    print(
        f"Prediction (Pos class proba) = {round(F.sigmoid(prediction).detach().item(), 3)} ({prediction})"
    )
    print(contributions.tolist())
    visuals.visualize_molecule_with_importance_gradient(
        smiles=smiles,
        contributions=contributions,
        threshold=threshold,
        scale_type=scale_type,
        func_groups=func_groups,
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
        return_class_as_proba: bool = False,
        return_true_targets: bool = False,
        desc="Predicting...",
        **kwargs,
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
            # print("Predicted output = ", output)

            if task in ["binary_classification"]:
                output = F.sigmoid(output)
                # print("bin output", output)
                if not return_class_as_proba:
                    threshold = kwargs.get("binary_threshold", 0.5)
                    # print(f"threshold={threshold}")
                    output = torch.as_tensor([int(x > threshold) for x in output]).to(
                        device
                    )
                pred_target = cat((pred_target, output), dim=0)

            elif task == "multiclass_classification":
                output = F.softmax(output)
                if return_class_as_proba:
                    output = F.softmax(output)
                else:
                    # _, predicted = torch.max(
                    #     output, 1
                    # )  ## returns maximum values(_), and their indices (predicted) along the dimmension 1
                    pred_target = cat((pred_target, output), dim=0)

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
        add_global_feats_to_nodes: bool = None,
        batch_size=128,
        device="cpu",
        shuffle=False,
        num_workers: int = 0,
        return_class_as_proba: bool = False,
        return_true_targets: bool = False,
        desc="Predicting...",
        **kwargs,
    ):
        if add_global_feats_to_nodes is None:
            add_global_feats_to_nodes = self.add_global_feats_to_nodes

        print("add_global_feats_to_nodes", add_global_feats_to_nodes)

        has_global_feats = True
        current_num_feats_x = None
        x_dim = self.model.in_channels

        if isinstance(data, List):
            has_global_feats = hasattr(data[0], "global_feats")
            current_num_feats_x = data[0].x.shape[1]

        else:
            has_global_feats = hasattr(data, "global_feats")
            current_num_feats_x = data.x.shape[1]

        # print("current_x_size = " {current_x_size})

        if add_global_feats_to_nodes and not has_global_feats:
            print(
                f"current_num_feats_x = {current_num_feats_x}  / x_dim = {x_dim}",
            )
            if current_num_feats_x == x_dim:
                warnings.warn(
                    "\n\n*******************************At least one data item does not have a global_feats. However, the x parameter has a size of {current_num_feats_x}, which is equal to the expected value of {x_dim}. So we suppose global features have been added already.\nThe global feature addition step will be skipped.\n*******************************"
                )
                add_global_feats_to_nodes = False
            elif current_num_feats_x < x_dim:
                raise Exception(
                    f"\n\n*******************************There are no features to add. The current x attribute of size {current_num_feats_x} indicates that no features were added to the original {x_dim} node features features."
                )
            else:
                raise Exception(
                    f"\n\n******************************* Current number of features ({current_num_feats_x}) exceeds the expected number ({x_dim})."
                )

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
            loader,
            device=device,
            return_class_as_proba=return_class_as_proba,
            return_true_targets=return_true_targets,
            desc=desc,
            **kwargs,
        )

    def predict_from_smiles_list(
        self,
        smiles_list: List[str],
        return_class_as_proba: bool = False,
        batch_size=128,
        device="cpu",
        shuffle=False,
        num_workers: int = 0,
        desc="Predicting...",
        **kwargs,
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
            return_class_as_proba=return_class_as_proba,
            batch_size=batch_size,
            device=device,
            shuffle=shuffle,
            num_workers=num_workers,
            desc=desc,
            **kwargs,
        )
