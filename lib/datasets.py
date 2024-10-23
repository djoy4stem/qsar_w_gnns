from random import shuffle
from typing import List
import math
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch import tensor, cat, stack, as_tensor, set_printoptions


def split_data(
    dataset: List[Data],
    split_indices: List[List[int]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = None,
    test_ratio: float = 0.2,
    shuffle_dataset: bool = True,
):
    train_data, val_data, test_data = None, None, None
    total_size = len(dataset)

    if split_indices is None:
        indices = list(range(total_size))
        # print(indices)
        if shuffle_dataset:
            shuffle(indices)
        # print(indices)

        train_size, val_size, test_size = 0, None, 0
        train_size = int(total_size * train_ratio)
        train_data = dataset[:train_size]

        if val_ratio is None:
            if train_ratio + test_ratio == 1:
                test_size = total_size - train_size
                test_data = dataset[train_size:]
            else:
                raise ValueError(
                    f"If val_ratio is None, train_ratio + val_ratio must be equals 1, not {train_ratio + test_ratio}."
                )
        else:
            assert math.isclose(
                0.9999999, 1, rel_tol=1e-06, abs_tol=1e-06
            ), f"train_ratio + val_ratio + test_ratio must be equals 1, not {train_ratio + val_ratio + test_ratio}."
            val_size = int(total_size * val_ratio)
            val_data = dataset[train_size : train_size + val_size]
            test_data = dataset[train_size + val_size :]

    elif len(split_indices) == 2:
        train_data = dataset[split_indices[0]]
        test_data = dataset[split_indices[1]]
    else:
        train_data = dataset[split_indices[0]]
        val_data = dataset[split_indices[1]]
        test_data = dataset[split_indices[2]]

    return train_data, val_data, test_data


def standardize_global_feats(batch, feature_scaler=StandardScaler()):
    gf = batch.global_feats
    # print(gf[:2])
    # feature_scaler = StandardScaler()
    gfs = feature_scaler.fit_transform(gf)
    batch.global_feats = gfs
    # print(gfs[:2])


# def merge_global_and_node_features(x:tensor, global_features:tensor):
#     # print(cat((x, cat([global_features.view(1,-1)] * x.shape[0])), dim=1))
#     return cat((x, cat([global_features.view(1,-1)] * x.shape[0])), dim=1)


# def batch_merge_global_to_node_features(batch:Batch):
#     # mols = []
#     for i in range(batch.batch_size):
#         # molx = batch[i].x
#         # print(tensor(batch[i].global_feats).unsqueeze(0))
#         # print(batch[i].__class__)
#         mol_global_features = tensor(batch[i].global_feats).unsqueeze(0)
#         # print(merge_global_and_node_features(molx, mol_global_features))
#         batch[i].x = merge_global_and_node_features(batch[i].x, mol_global_features)
#         # print(f'batch[{i}].x = ', batch[i].x)
#         # print(help(batch[i].update_tensor))
#         return batch
#         # mols.append(Data(merge_global_and_node_features(molx, mol_global_features)))
#         # print(mols[0].__class__)
#         # return Batch.from_data_list(mols)


# def get_dataloader(dataset:List[Data], batch_size=128, shuffle:bool=False
#                    , add_global_feats_to_nodes:bool=False):

#     dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
#     if not dataset[0].global_feats is None:
#         for batch in dl:
#             # print(type(batch), dir(batch))
#             standardize_global_feats(batch)
#             if add_global_feats_to_nodes:
#                 batch = batch_merge_global_to_node_features(batch)

#     return dl


def clean_features(features: List[List], feature_scaler=MinMaxScaler()):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    stdzr = feature_scaler

    cleaned = imputer.fit_transform(features)
    # print("imputed\n", imputed)
    if not feature_scaler is None:
        cleaned = stdzr.fit_transform(cleaned)

    return cleaned


import time


def get_dataloader(
    dataset: List[Data],
    batch_size=128,
    shuffle: bool = False,
    add_global_feats_to_nodes: bool = False,
    num_workers: int = 0,
    scale_features: bool = True,
    feature_scaler=MinMaxScaler(),
):
    new_dataset = []
    x_cleaned, edge_attr_cleaned, gfeatures_cleaned = [], [], []
    has_edge_attr = (
        hasattr(dataset[0], "edge_attr") and not dataset[0].edge_attr is None
    )
    has_global_feats = (
        hasattr(dataset[0], "global_feats") and not dataset[0].global_feats is None
    )

    for d in dataset:
        # print(f"d.x[1][:11] ({d.x.__class__.__name__}): {d.x[1][:11]}")
        # set_printoptions(profile="full")
        # print(f"d.x ({d.x.shape}): {d.x}")
        # set_printoptions(profile="default")
        x_cleaned.extend(d.x.tolist())
        # print(len(x_cleaned))
        # print(f"x_cleaned[1][:11] : {x_cleaned[1][11]}")

        if has_edge_attr:
            edge_attr_cleaned.extend(d.edge_attr.tolist())

        if has_global_feats:
            # print("====> has_global_feats")
            # gfeatures_cleaned = [i.global_feats.tolist() for i in dataset]
            # print(d.global_feats.tolist())
            gfeatures_cleaned.append(d.global_feats.tolist())

    # print("len(gfeatures_cleaned) = ", len(gfeatures_cleaned))
    # print('gfeatures_cleaned', len(gfeatures_cleaned), gfeatures_cleaned[:2])

    # print('dataset[0].x before cleaning', dataset[0].x[0])
    if scale_features:
        # x_cleaned =  clean_features(features = x_cleaned, feature_scaler=feature_scaler)
        # # print('dataset[0].x after cleaning', x_cleaned[0], x_cleaned[0].shape)
        pass

    if not len(edge_attr_cleaned) == 0:
        if scale_features:
            # # print('dataset[0].edge_attr before cleaning', dataset[0].edge_attr[0])
            # edge_attr_cleaned = clean_features(features=edge_attr_cleaned, feature_scaler=feature_scaler)
            # # print('dataset[0].edge_attr after cleaning', edge_attr_cleaned[0], edge_attr_cleaned[0].shape)
            # # print(f"Num. edge_attr. cleaned = {edge_attr_cleaned.shape}")
            pass
    else:
        edge_attr_cleaned = None
        # print(edge_attr_cleaned[0:73])

    if not len(gfeatures_cleaned) == 0:
        # print("gfeatures_cleaned = ", gfeatures_cleaned[:2])
        if scale_features:
            # print('gfeatures before cleaning', gfeatures_cleaned[:2])
            gfeatures_cleaned = clean_features(
                features=gfeatures_cleaned, feature_scaler=feature_scaler
            )
            print("gfeatures_cleaned", gfeatures_cleaned.shape)
            # print('gfeatures after cleaning',gfeatures_cleaned[:2])
    else:
        gfeatures_cleaned = None

    curr_atom_index = 0
    for i in range(len(dataset)):
        curr_d = dataset[i].clone()
        
        curr_num_atoms = dataset[i].x.shape[0]
        # print(i+1, 'atoms: ', curr_atom_index, curr_num_atoms)
        curr_d.x = tensor(x_cleaned[curr_atom_index : curr_atom_index + curr_num_atoms])
        curr_atom_index = curr_num_atoms

        if has_edge_attr:
            curr_bond_index = 0
            curr_num_bonds = dataset[i].edge_attr.shape[0]
            # print(i+1, 'bonds: ', curr_bond_index, curr_num_bonds, edge_attr_cleaned.shape)
            curr_d.edge_attr = tensor(
                edge_attr_cleaned[curr_bond_index : curr_bond_index + curr_num_bonds]
            )
            curr_bond_index = curr_num_bonds

        curr_dict = curr_d.to_dict()
        # print(curr_dict)

        if (
            add_global_feats_to_nodes
            and not curr_dict.get("global_feats", None) is None
        ):
            # print(cat([g.view(1,-1)] * d.shape[0]))
            # print("d.x.shape: Before adding global = ", d.x.shape)
            # print("global features", gfeatures_cleaned[i].shape)
            # print(d.x)
            # print(type(gfeatures_cleaned[i]))
            # print(gfeatures_cleaned[i])
            # print(gfeatures_cleaned[i].view(1,-1))
            # print('gfeatures_cleaned', gfeatures_cleaned[i])
            curr_d.global_feats = as_tensor(
                gfeatures_cleaned[i]
            )  ## using as_tensor (or from_numpy) converts the numpy array to a tensor WITHOUT COPYING the data
            my_stack = stack([as_tensor(curr_d.global_feats)] * curr_d.x.shape[0])
            # print(my_stack)
            # time.sleep(5)
            curr_d.x = cat((curr_d.x, my_stack), dim=1)
            # print("d.x.shape: After adding global = ", d.x.shape)
            new_dataset.append(curr_d)
            curr_d.global_feats = None  # We make sure that once the concatenation occurs, we set the global_feats to None. Helps avoiding confusion.
        elif "global_feats" in curr_dict and not curr_d.global_feats is None:
            curr_d.global_feats = as_tensor(gfeatures_cleaned[i]).view(1, -1)
            new_dataset.append(curr_d)
            # print('curr_d global feats', curr_d.global_feats)
        elif not "global_feats" in curr_dict:
            new_dataset.append(curr_d)

    pin_memory = False
    if num_workers > 0:
        pin_memory = True

    return DataLoader(
        dataset=new_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def get_dataloaders(
    train_data: List[Data],
    test_data: List[Data],
    val_data: List[Data] = None,
    batch_size: int = 128,
    shuffle_train: bool = False,
    add_global_feats_to_nodes: bool = False,
    num_workers: int = 0,
    scale_features: bool = True,
    feature_scaler=MinMaxScaler(),
):
    train_dataloader = get_dataloader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=shuffle_train,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
        scale_features=scale_features,
        feature_scaler=feature_scaler,
    )

    test_dataloader = get_dataloader(
        dataset=test_data,
        batch_size=batch_size,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
        scale_features=scale_features,
        feature_scaler=feature_scaler,
    )

    if val_data is None:
        return train_dataloader, test_dataloader
    else:
        val_dataloader = get_dataloader(
            dataset=val_data,
            batch_size=batch_size,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            num_workers=num_workers,
            scale_features=scale_features,
            feature_scaler=feature_scaler,
        )
        return train_dataloader, val_dataloader, test_dataloader
