from random import shuffle
from typing import List
import math
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data, Batch

# from torch_geometric.loader import DataLoader

## torch.utils.data with collate_fn and Collater. It seems PyG removed collate_fn and replaced with Collater.
# but I could not get the proper behavior
from torch.utils.data import DataLoader
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


class CustomBatch(Batch):
    @staticmethod
    def from_data_list(data_list):
        
        # Create the batch using the parent method
        batch = super(CustomBatch, CustomBatch).from_data_list(data_list)

        if hasattr(batch, 'global_feats'):
            # print(sum([d.global_feats.shape[0] for d in data_list]))
            # Concatenate all `global_feats` attributes as a 2D tensor
            if data_list[0].global_feats.ndim>1:
                global_feats = cat([data.global_feats for data in data_list], dim=0)
            elif data_list[0].global_feats.ndim==1:
                global_feats = cat([data.global_feats.unsqueeze(0) for data in data_list], dim=0)
            # print("GF", global_feats.shape)

            # Add the `global_feats` as a 2D tensor
            batch.global_feats = global_feats
        return batch

## Custom PyTorch Collate Function: https://lukesalamone.github.io/posts/custom-pytorch-collate/
class CustomCollater:
    def __init__(self, follow_batch=[]):
        self.follow_batch = follow_batch

    def __call__(self, batch):
        # Stack and batch standard attributes with PyG's Batch class
        batch_graph = Batch.from_data_list(batch, follow_batch=self.follow_batch)
        # print("batch_graph", batch_graph)
        # Handle `global_feats` if it exists
        if "global_feats" in batch[0]:
            global_feats = stack([data.global_feats for data in batch])
            batch_graph.global_feats = global_feats
        else:
            batch_graph.global_feats = None

        return batch_graph


def get_dataloader(
    dataset: List[Data],
    batch_size=128,
    shuffle: bool = False,
    add_global_feats_to_nodes: bool = False,
    num_workers: int = 0,
    colllater: CustomCollater = CustomCollater(),
):
    if add_global_feats_to_nodes:
        dataset = gnn_utils.add_global_features_to_nodes(data=dataset)
        print("dataset after adding global feat: ", dataset)
    # elif has_attr(dataset, 'global_feats'):

    pin_memory = False
    if num_workers > 0:
        pin_memory = True

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=colllater,
    )


class CustomCollater:
    def __init__(self, follow_batch=[]):
        self.follow_batch = follow_batch

    def __call__(self, batch):
        # Stack and batch standard attributes with PyG's Batch class
        batch_graph = Batch.from_data_list(batch, follow_batch=self.follow_batch)

        # Handle `global_feats` if it exists
        if "global_feats" in batch[0]:
            global_feats = stack([data.global_feats for data in batch])
            batch_graph.global_feats = global_feats
        else:
            batch_graph.global_feats = None  # None if not present

        return batch_graph


def get_dataloaders(
    train_data: List[Data],
    test_data: List[Data],
    val_data: List[Data] = None,
    batch_size: int = 128,
    shuffle_train: bool = False,
    add_global_feats_to_nodes: bool = False,
    num_workers: int = 0,
):
    custom_collater = CustomCollater()
    train_dataloader = get_dataloader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=shuffle_train,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
        colllater=custom_collater,
    )

    test_dataloader = get_dataloader(
        dataset=test_data,
        batch_size=batch_size,
        add_global_feats_to_nodes=add_global_feats_to_nodes,
        num_workers=num_workers,
        colllater=custom_collater,
    )

    if val_data is None:
        return train_dataloader, test_dataloader
    else:
        val_dataloader = get_dataloader(
            dataset=val_data,
            batch_size=batch_size,
            add_global_feats_to_nodes=add_global_feats_to_nodes,
            num_workers=num_workers,
            colllater=custom_collater,
        )
        return train_dataloader, val_dataloader, test_dataloader
