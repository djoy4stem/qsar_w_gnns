from matplotlib import pyplot as plt

import os
from os.path import join
import sys
from IPython.display import display
from PIL import Image

ROOT_DIR = os.sep.join(os.path.abspath(".").split(os.sep)[:-1])
sys.path.insert(0, ROOT_DIR)
DATASET_DIR = "{}/data".format(ROOT_DIR)
print(DATASET_DIR)

from typing import List, Union, Any, Tuple
from datetime import datetime
from random import sample

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Data
from torch.nn import BCELoss, LeakyReLU, ReLU
from torch.optim import lr_scheduler, Adagrad, Adadelta, Adam, AdamW

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    Draw,
    AllChem,
    PandasTools,
    MolFromSequence,
    MolToSmiles,
    MolToInchiKey,
    Descriptors,
    GraphDescriptors,
)
from rdkit.Chem import rdMolDescriptors as rdmdesc

from joblib import Parallel, delayed

# RDLogger.DisableLog('rdApp.*')
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

from lib import (
    gnn_utils,
    utilities,
    datasets,
    splitters,
    featurizers,
    training_utils,
    graph_nns,
    graph_utils,
    gnn_explainer,
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from time import time


dd2_dataset_fname = f"{DATASET_DIR}/dd2_datasets.pt"
dd2_dataset = torch.load(dd2_dataset_fname)

train_dataset = dd2_dataset["train_dataset"]
val_dataset = dd2_dataset["val_dataset"]
test_dataset = dd2_dataset["test_dataset"]

dd2_loader_fname = f"{DATASET_DIR}/dd2_dataloaders.pt"
dd2_loaders = torch.load(dd2_loader_fname)

train_loader = dd2_loaders["train_loader"]
val_loader = dd2_loaders["val_loader"]
test_loader = dd2_loaders["test_loader"]


gcn_predictor = torch.load(f"{DATASET_DIR}/models/dd2_class_gcn_predictor.pt")


smiles = ["CCC(=O)NCc1nccc1CO", "O=C(O)c1cccnc1Sc1cc(CCO)c(Cl)cc1"]
new_preds = gcn_predictor.predict_from_smiles_list(
    smiles_list=smiles, device="cuda:0", desc="Predicting..."
)

threshold = 0.5
pred_classes = [int(x > threshold) for x in new_preds.squeeze(1)]
print("\nnew_preds", new_preds.detach())
print("pred_classes", pred_classes)


features_mapping = gcn_predictor.atom_featurizer.features_mapping
print(features_mapping["atomic_num"])

node_dict = {
    features_mapping["atomic_num"][i]: Chem.Atom(i).GetSymbol()
    if not i == "unk"
    else "unk"
    for i in features_mapping["atomic_num"]
}
print("node_dict = ", node_dict)


gnn_xp_module = gnn_explainer.GNNExplainerModule(
    gnn_predictor=gcn_predictor,
    node_dict=node_dict,
    num_epochs=1000,
    lr=0.005,
    node_mask_type="attributes",
    edge_mask_type="object",
    explanation_type="model",
    model_config=dict(
        mode="binary_classification", task_level="graph", return_type="raw"
    ),
    node_colors=None,
    device="cuda:0",
)

print(gnn_xp_module)


subgraph_size = 5


print("Training subgraph based logreg predictor")
t0 = time()

##Train model
gnn_xp_module.train_subgraph_based_logreg_predictor(
    train_dataset[:6000], subgraph_size=subgraph_size
)

t1 = time()
print(f"Trained subgraph based logreg predictor in {t1-t0} seconds.")

## Evaluate model
gnn_xp_module.evaluate_subgraph_based_logreg_predictor(test_dataset[:])
t2 = time()
print(f"Trained subgraph based logreg predictor in {t2-t1} seconds.")


torch.save(gnn_xp_module, f"{DATASET_DIR}/models/dd2_class_gcn_explanation_module.pt")
