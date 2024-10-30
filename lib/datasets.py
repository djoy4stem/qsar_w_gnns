from random import shuffle
from typing import List
import math
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch import tensor, cat, stack, as_tensor, set_printoptions
import time

from lib import gnn_utils


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




def get_dataloader(
    dataset: List[Data],
    batch_size=128,
    shuffle: bool = False,
    add_global_feats_to_nodes: bool = False,
    num_workers: int = 0,    

):
    if add_global_feats_to_nodes:
        dataset = gnn_utils.add_global_features_to_nodes(data=dataset)
        print("dataset after adding global feat: ", dataset)
    
    pin_memory = False
    if num_workers > 0:
        pin_memory = True

    return DataLoader(
        dataset=dataset,
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
):
    train_dataloader = get_dataloader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=shuffle_train,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
    )

    test_dataloader = get_dataloader(
        dataset=test_data,
        batch_size=batch_size,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
    )

    if val_data is None:
        return train_dataloader, test_dataloader
    else:
        val_dataloader = get_dataloader(
            dataset=val_data,
            batch_size=batch_size,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            num_workers=num_workers
        )
        return train_dataloader, val_dataloader, test_dataloader
