from os import getcwd, makedirs
from os.path import join, exists, isdir
from shutil import rmtree
from sklearn.metrics import r2_score, roc_auc_score, f1_score, mean_squared_error
from sklearn.metrics._scorer import _PredictScorer
import torch.optim as optim
from torch.optim import *
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import *
from torch import tensor, argmax
from torch.nn import (
    MSELoss,
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    BCELoss,
    ReLU,
    functional as F,
)
from torch.utils.data import DataLoader
from torch_geometric.loader import NodeLoader, LinkLoader
from torch_geometric.data import Data
from torch.nn.utils import clip_grad_norm_
import torch

from torch.utils.tensorboard import SummaryWriter

from typing import Optional, Any, Dict, Union
import numpy as np
from tqdm import tqdm
import warnings
import inspect
import ast

from typing import List


from optuna.samplers import BaseSampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from optuna import create_study
from optuna.trial import TrialState

import sklearn
from sklearn import utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from lib import graph_nns, datasets, predictions
from lib.utilities import check_if_param_used, compute_score

SCORING_FUNC = {
    "regression": {"r2": r2_score, "mse": mean_squared_error},
    "binary_classification": {"roc_auc": roc_auc_score, "f1_score": f1_score},
    "multilabel_classification": {"ro_auc": roc_auc_score, "f1_score": f1_score},
    "multiclass_classification": {"ro_auc": roc_auc_score, "f1_score": f1_score},
}


class GNNTrainer(object):
    def __init__(
        self,
        task,
        model,
        scheduler=None,
        optimizer=None,
        criterion=None,
        scoring_func=None,
        n_epochs=25,
        learning_rate=1e-3,
        save_dir: str = None,
        resume: bool = False,
        # tboard_writer: SummaryWriter  = None,
        **kwargs
    ):
        self.task          = task
        self.model         = model
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        # self.tboard_writer  = tboard_writer

        if isinstance(optimizer, str):
            optimizer_params_dict = ast.literal_eval(optimizer)
            self.create_optimizer_from_dict(optimizer_params_dict)
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.0,
                weight_decay=1e-3,
            )
            # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)

        print("\toptimizer: ", self.optimizer.state_dict())

        if isinstance(scheduler, str):
            scheduler_params_dict = ast.literal_eval(scheduler)
            self.create_lr_scheduler_from_dict(scheduler_params_dict)
        else:
            ## The default scheduler is of type ReduceOnPlateau, which monitors a specified metric (e.g., validation loss) and reduces
            # the learning rate when the metric stops improving for a specified number of steps.
            # self.scheduler = scheduler if scheduler is not None else lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer
            #                                                             , mode='min' # reduce the learning rate when the monitored metric (e.g., validation loss) stops decreasing
            #                                                             , factor=0.7
            #                                                             , patience=30 # Number of epochs with no improvement after which learning rate will be reduced
            #                                                             , verbose=True
            #                                                             , threshold=5e-02
            #                                                             , threshold_mode='rel', cooldown=0
            #                                                             , min_lr=self.learning_rate*0.1
            #                                                             , eps=1e-08)

            self.scheduler = (
                scheduler
                if scheduler is not None
                else lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=self.learning_rate,
                    max_lr=0.2,
                    step_size_up=100,
                    cycle_momentum=True,
                )
            )
            # self.scheduler = scheduler if scheduler is not None else lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        self.scheduler_step_requires_metrics = (
            "metrics" in inspect.signature(self.scheduler.step).parameters
        )

        print("\tlr_scheduler: ", self.scheduler.state_dict())

        self.criterion = criterion
        if self.criterion is None:
            if self.task == "regression":
                self.criterion = MSELoss()
            elif self.task in ["binary_classification", "multilabel_classification"]:
                self.criterion = BCEWithLogitsLoss()
            elif self.task == "nulticlass_classification":
                self.criterion = CrossEntropyLoss()

        self.scoring_func = scoring_func
        if self.scoring_func is None:
            if self.task == "binary_classification":
                self.scoring_func = SCORING_FUNC["binary_classification"]["roc_auc"]
            elif self.task == "multiclass_classification":
                self.scoring_func = SCORING_FUNC["multiclass_classification"]["roc_auc"]
            elif self.task == "regression":
                self.scoring_func = SCORING_FUNC["regression"]["r2"]

        elif isinstance(self.scoring_func, str):
            self.scoring_func = eval(f"{scoring_func}")

        # print('scoring_func=', self.scoring_func)
        self.save_dir = save_dir
        self.checkpoint_dir = None
        if not (self.save_dir is None or resume):
            self.checkpoint_dir = join(save_dir, "checkpoints")
            if exists(self.checkpoint_dir) and isdir(self.checkpoint_dir):
                rmtree(self.checkpoint_dir)
            makedirs(self.checkpoint_dir)

        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"checkpoint dir: {self.checkpoint_dir}")
        # print(f"scheduler: {dir(self.scheduler)}")
        # print(self.scheduler.state_dict())

    @classmethod
    def from_dict(cls, params_grid: dict):
        task = params_grid.get("task")
        model = params_grid.get("model")
        learning_rate = params_grid.get("learning_rate", 1e-3)
        n_epochs = params_grid.get("n_epochs", 25)
        # print('model', model)
        optimizer = params_grid.get("optimizer", None)

        criterion = params_grid.get("criterion", None)
        scheduler = params_grid.get("scheduler", None)
        scoring_func = params_grid.get("scoring_func", None)
        save_dir = params_grid.get("save_dir", None)
        resume = params_grid.get("resume", False)
        tboard_writer = params_grid.get("tboard_writer", None)

        return cls(
            task=task,
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            criterion=criterion,
            scoring_func=scoring_func,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            save_dir=save_dir,
            resume=resume,
            tboard_writer=tboard_writer
        )

    def create_optimizer_from_dict(self, params_dict):
        if not "optimizer_type" in params_dict:
            raise ValueError(
                "The optimizer_type is missing. Please add a valid type for the optimizer. e.g.: torch.optim.SGD."
            )

        parameters_ = list(
            inspect.signature(eval(params_dict["optimizer_type"])).parameters.keys()
        )
        params_dict["params"] = self.model.parameters()
        params_dict["lr"] = self.learning_rate
        new_pd = {}
        for pm_ in params_dict:
            if pm_ in parameters_:
                new_pd[pm_] = params_dict[pm_]
        self.optimizer = eval(params_dict["optimizer_type"])(**new_pd)

    def create_lr_scheduler_from_dict(self, params_dict):
        if not "lr_scheduler_type" in params_dict:
            raise ValueError(
                "The lr_scheduler_type is missing. Please add a valid type for the optimizer. e.g.: lr_scheduler.StepLR."
            )

        # print('eval(params_dict["lr_scheduler_type"])', eval(params_dict["lr_scheduler_type"]))
        # print('Is ReduceLROnPlateau', eval(params_dict["lr_scheduler_type"]) == torch.optim.lr_scheduler.ReduceLROnPlateau)
        # print('Is CyclicLR', eval(params_dict["lr_scheduler_type"]) == torch.optim.lr_scheduler.CyclicLR)

        parameters_ = list(
            inspect.signature(eval(params_dict["lr_scheduler_type"])).parameters.keys()
        )
        params_dict["optimizer"] = self.optimizer
        new_pd = {}
        for pm_ in params_dict:
            if pm_ in parameters_:
                new_pd[pm_] = params_dict[pm_]

        if (
            eval(params_dict["lr_scheduler_type"])
            == torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            # print("\nOK. This is ReduceLROnPlateau")
            min_lr = params_dict.get("max_lr", self.learning_rate * 0.01)
            if min_lr > self.learning_rate:
                warnings.warn(
                    f"For 'ReduceOnPlateau' schedulers, the min_lr must be lower than the learning rate. The provided value was {min_lr} (> {self.learning_rate}). We will thus set min_lr={self.learning_rate*0.01}"
                )
                min_lr = self.learning_rate * 0.01
                new_pd["min_lr"] = min_lr

        elif (
            eval(params_dict["lr_scheduler_type"]) == torch.optim.lr_scheduler.CyclicLR
        ):
            # print("\nOK. This is CycleLR")
            new_pd["base_lr"] = self.learning_rate
            max_lr = new_pd.get("max_lr", self.learning_rate + 0.2)
            if max_lr < self.learning_rate:
                warnings.warn(
                    "The provided max_lr is smaller than min_lr. We will set max_lr=self.learning_rate+0.2"
                )
                warnings.warn(
                    f"For 'CyclicLR' schedulers, the max_lr must be greater than the learning rate. The provided value was {max_lr} (< {self.learning_rate}). We will thus set min_lr={self.learning_rate+0.2}"
                )

                max_lr = self.learning_rate + 0.2
                new_pd["max_lr"] = max_lr

        self.scheduler = eval(params_dict["lr_scheduler_type"])(**new_pd)

    def train(self, train_loader, val_loader, save_every=0, n_epochs=None, device=None, trial=None, tboard_writer:SummaryWriter=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = 'cpu'
        self.model.to(device)

        print(f"device = {device}")
        print(f"TASK: {self.task}")
        # print(f"MODEL: {self.model}")

        val_scores, val_losses, train_losses = [], [], []
        if n_epochs is None:
            n_epochs = self.n_epochs
        for epoch in range(n_epochs):
            # print(f"\n ************************************** epoch {epoch}")
            train_loss = self.train_epoch(self.model, train_loader, device)
            train_losses.append(train_loss)

            val_loss, val_score = self.validate_epoch(self.model, val_loader, device)
            val_losses.append(val_loss)
            val_scores.append(val_score)

            self.learning_rate = self.optimizer.param_groups[0]["lr"]


            if not self.scheduler is None:
                # self.scheduler.step()
                self.scheduler_step(val=val_loss)

            if not trial is None:
                trial.report(val_loss, epoch)

                if not tboard_writer is None:
                    # Log metrics for each epoch to TensorBoard
                    tboard_writer.add_scalar(f"Trial_{trial.number}/Train_Loss", train_loss, epoch)
                    tboard_writer.add_scalar(f"Trial_{trial.number}/Val_Loss", val_loss, epoch)
                    tboard_writer.add_scalar(f"Trial_{trial.number}/Val_Score", val_score, epoch)
                    tboard_writer.add_scalar(f"Trial_{trial.number}/Learning_Rate", self.learning_rate, epoch)

                # Check if trial should be pruned based on intermediate results
                if trial.should_prune():
                    if not tboard_writer is None:
                        tboard_writer.add_text(f"Trial_{trial.number}/Status", "Pruned", epoch)
                    raise TrialPruned()






            if epoch % 20 == 0 or epoch == self.n_epochs - 1:
                print(
                    f"\n===> Epoch {epoch + 1:4d}/{self.n_epochs}: Average Train Loss: {train_loss:.3f} |  Average validation Loss: {val_loss:.3f} | Validation Score: {val_score:.3f} | lr: {self.learning_rate:.5f}"
                )
                # print(f"Scheduler = {self.scheduler.state_dict()}")

            # print(epoch, save_every> 0, (not self.checkpoint_dir is None), epoch+1%save_every==0, epoch == self.n_epochs-1)
            if (
                save_every > 0
                and (not self.checkpoint_dir is None)
                and (epoch + 1 % save_every == 0 or epoch == self.n_epochs - 1)
            ):
                print(
                    f"Saving model at epoch {epoch+1} to {join(self.checkpoint_dir, f'ckeckpoint_{epoch+1}.ckpt')}"
                )
                save_ckpt(
                    model=self.model,
                    file_path=join(self.checkpoint_dir, f"ckeckpoint_{epoch+1}.ckpt"),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )

        return train_losses, val_losses, val_scores

    def train_epoch(self, model, train_loader, device):
        model.train()
        train_losses = []
        batch_nr = 0

        # for batch in tqdm(train_loader):
        for batch in train_loader:
            batch_nr += 1

            batch = batch.to(device)
            output, train_true = predictions.predict_from_batch(
                batch=batch, model=model, return_true_targets=True
            )
            # print(f"\noutput[:5] = {output[:5]}\ntrain_true[:5] = {train_true[:5]}")
            train_loss = None
            # print("\n", f"output[:5] = {output[:5]}\n") #, f"torch.max = {torch.max(output[:5], 1)}
            # print(f"self.task = {self.task} ({self.task == 'binary_classification'})")

            if self.task == "binary_classification":
                # print(f"\nComputing ({self.task}) Loss with {self.criterion}\n")
                train_loss = self.compute_loss(self.criterion, output, train_true)
                # print("train loss = ", train_loss)

            elif self.task == "multiclass_classification":
                train_loss = self.compute_loss(
                    self.criterion, output, train_true.long()
                )

            elif self.task == "regression":
                train_loss = self.compute_loss(self.criterion, output, train_true)
                # print(f"output = {output.shape}", "\n", f"train_true = {train_true.view(-1,1).shape}")

            else:
                raise ValueError(f"No implementation for task {self.task}.")

            # print(f"loss is None: {loss is None}")
            if not train_loss is None:
                # print(f"train_true = {train_true.view(-1,1).shape}\nCriterion: {self.criterion}\nTrain Loss = {loss}")
                train_losses.append(train_loss.item())
                train_loss.backward()
            else:
                print(
                    f"There was an issue computing the loss for training batch {batch_nr}. The loss was set to None, and will not be considered for averaging."
                )

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()  ## Update parameters based on gradients.
            self.optimizer.zero_grad()  ## Clear gradients.

        mean_loss = np.array(train_losses).mean()

        return mean_loss

    def validate_epoch(self, model, val_loader, device, threshold=0.5):
        model.eval()
        scores = []
        losses = []
        batch_counter = 0
        for batch in val_loader:
            batch = batch.to(device)
            batch_counter += 1
            val_pred, val_true = predictions.predict_from_batch(
                batch=batch, model=model, return_true_targets=True
            )

            val_loss = None
            # print("val_pred = ", val_pred)
            if self.task in [
                "binary_classification",
                "multiclass_classification",
                "regression",
            ]:
                val_loss = self.compute_loss(self.criterion, val_pred, val_true)
                # print(f"val_loss = {val_loss}")
                val_score = self.compute_score(
                    self.scoring_func,
                    pred_target=val_pred,
                    true_target=val_true,
                    task=self.task,
                )
                # print(f"\nvalidation score = {val_score}")
                scores.append(val_score)
            else:
                raise ValueError(f"No implementation for task {self.task}.")

            # print(f"train_true = {train_true.view(-1,1).shape}")
            # print(f"Criterion: {self.criterion}")
            # print(f"Val Loss = {loss}")

            if not val_loss is None:
                losses.append(val_loss.item())
                val_loss.backward()
            else:
                print(
                    f"There was an issue computing the loss for validation batch {batch_counter}. The loss was set to None, and will not be considered for averaging."
                )
            # print('self.scoring_func', self.scoring_func)

        mean_loss = np.array(losses).mean()
        mean_score = np.array(scores).mean()

        return mean_loss, mean_score

    def scheduler_step(self, val: float = None):
        if self.scheduler_step_requires_metrics:
            self.scheduler.step(metrics=val)
        else:
            self.scheduler.step()

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def compute_loss(self, criterion, pred_target, true_target, task=None):
        task = task or self.task
        # print(f"task = {task}")
        if True:
            # try:
            if task == "regression":
                return self.criterion(pred_target, true_target.view(-1, 1))

            elif task == "binary_classification":
                if isinstance(criterion, BCEWithLogitsLoss):
                    # print(f"self.criterion(pred_target.float(), true_target) = {self.criterion(pred_target.float(), true_target)}")
                    return self.criterion(pred_target.squeeze(1).float(), true_target)
                elif isinstance(criterion, BCELoss):
                    return self.criterion(
                        F.sigmoid(pred_target).squeeze(1).float(), true_target
                    )

            elif task == "multiclass_classification":
                if not true_target.dtype == torch.long:
                    true_target = true_target.long()
                return self.criterion(pred_target.squeeze(1).float(), true_target)

            else:
                raise ValueError(f"Task ({task}) not supported for loss computation.")
        # except Exception as exp:
        #     return None

    def compute_score(self, scoring_func, pred_target, true_target, task=None):
        task = task or self.task
        return compute_score(scoring_func, pred_target, true_target, task)


def save_ckpt(
    model: torch.nn.Module,
    file_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
):
    r"""Saves the model checkpoint at a given epoch."""
    ckpt: Dict[str, Any] = {}
    ckpt["model_state"] = model.state_dict()
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["sheculer_state"] = scheduler.state_dict()

    torch.save(ckpt, file_path)


def modules_are_equal(module_1, module_2):
    state_dict1 = module_1.state_dict()
    state_dict2 = module_2.state_dict()
    print(f"state_dict1 keys = {list(state_dict1.keys())}")
    print(f"state_dict2 keys = {list(state_dict2.keys())}")

    is_equal = True
    for key in state_dict1:
        if key in state_dict2:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                is_equal = False

    return is_equal



class OptunaHPO:
    def __init__(self, n_trials:int=5, n_jobs:int=-1, sampler=None
                , n_startup_trials:int =1 ## The number of initial trials that won’t be pruned. This helps build up a baseline of performance
                , n_warmup_steps:int =25  ## The minimum number of reporting steps (i.e.: epochs in this implementation, and not batches.) before a trial can be pruned.
                , add_tboard_writer:bool = False, tboard_log_dir:str="logs/optuna_hyperparameter_tuning"
            ):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.best_model = None
        self.best_val_score = None
        self.train_val_metadata = {
            "train_losses": None,
            "val_losses": None,
            "val_scores": None,
        }
        self.n_startup_trials=n_startup_trials
        self.n_warmup_steps=n_warmup_steps
        self.add_tboard_writer = add_tboard_writer
        if self.add_tboard_writer:
            # Initialize TensorBoard SummaryWriter
            if tboard_log_dir is None:
                warnings.warn("The log destination was set to None. It will be set to 'logs/optuna_hyperparameter_tuning' in order to write a summary.")
                self.tboard_log_dir = "logs/optuna_hyperparameter_tuning"
            else:
                self.tboard_log_dir = tboard_log_dir
            self.tboard_writer = SummaryWriter(log_dir=self.tboard_log_dir)
            self.tboard_log_dir = tboard_log_dir
        else:
            self.tboard_writer = None
            self.tboard_log_dir=tboard_log_dir


    def get_params(self):
        return {
            "n_trials": self.n_trials,
            "n_jobs": self.n_jobs,
            "n_startup_trials": self.n_startup_trials,
            "n_warmup_steps": self.n_warmup_steps,
            "sampler": self.sampler,
            "best_model": self.best_model,
            "best_val_score": self.best_val_score,
        }
 

    def train_and_validate(
        self,
        trial,
        # gnn_trainer: GNNTrainer,
        params: dict,
        dataloaders: List[DataLoader],
        split_mode: str = "classic",
        # add_tboard_writer: bool = False
    ):
        if split_mode == "classic":
            gnn_params = {
                p: params[p]
                for p in params
                if check_if_param_used(eval(str(params["model"])), p)
            }

            gnn_model = eval(f"{str(params['model'])}.from_dict({gnn_params})")

            # task = params['task']
            # params = gnn_params
            # params['task'] = task
            params["model"] = gnn_model
            gnn_trainer = GNNTrainer.from_dict(params)

            train_losses, val_losses, val_scores = gnn_trainer.train(
                trial=trial,
                train_loader=dataloaders[0],
                val_loader=dataloaders[1],
                save_every=0,
                n_epochs=None,
                tboard_writer=self.tboard_writer
            )

            val_score = val_scores[-1]
            # print(f"VAL SCORE = {val_score}")

            if self.best_val_score is None:
                self.best_model = gnn_trainer.model
                self.best_val_score = val_score
                self.train_val_metadata = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_scores": val_scores,
                }
                # print("best_val_score = ", self.best_val_score)
            elif val_score > self.best_val_score:
                print(
                    f"\n*** We have a better model with val_score = {val_score} (> {self.best_val_score}) ***\n"
                )
                print(
                    f"modules_are_equal ={modules_are_equal(self.best_model,gnn_trainer.model)}\n"
                )
                self.best_model = gnn_trainer.model
                self.best_val_score = val_score
                self.train_val_metadata = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_scores": val_scores,
                }

            return val_score
            # return val_losses[-1]
        else:
            raise NotImplementedError(
                "train_validate is only implemented for the 'classic' split mode."
            )


    def objective(
        self,
        # task:int,
        # gnn_model:torch.nn.Module,
        trial,
        train_val_data: Union[List[DataLoader], List[List[Data]]],
        params_grid: dict,
        split_mode: str = "classic",
        **kwargs,
    ):
        params = pick_params(trial, params_grid)
        print(f"params = {params}\n")

        dataloaders = train_val_data
        # print(f"dataloaders = {dataloaders}")
        # print(f"train_val_data[0] = {train_val_data[0]}")
        # if not isinstance(train_val_data[0], (DataLoader, NodeLoader, LinkLoader)) and isinstance(
        #     train_val_data[0][0], Data
        # ):
        first_batch = next(iter(train_val_data[0]))
        if not isinstance(
            train_val_data[0], (DataLoader, NodeLoader, LinkLoader)
        ) and isinstance(first_batch, Data):
            dl_params = {}
            batch_size = params.get("batch_size", 128)
            shuffle = params.get("shuffle", False)
            add_global_feats_to_nodes = params.get("add_global_feats_to_nodes", False)
            num_workers = params.get("num_workers", 0)
            # standardize = params.get("standardize", True)
            # standardizer = params.get("standardizer", MinMaxScaler())

            print("Creating DataLoader objects...")
            train_loader = datasets.get_dataloader(
                dataset=train_val_data[0],
                batch_size=batch_size,
                shuffle=shuffle,
                add_global_feats_to_nodes=add_global_feats_to_nodes,
                num_workers=num_workers,
            )
            val_loader = datasets.get_dataloader(
                dataset=train_val_data[1],
                batch_size=batch_size,
                shuffle=shuffle,
                add_global_feats_to_nodes=add_global_feats_to_nodes,
                num_workers=num_workers,
                # standardize=standardize,
                # standardizer=standardizer,
            )
            dataloaders = [train_loader, val_loader]

        params["in_channels"] = first_batch.x[0].shape[0]
        # params.update(**{'in_channels' : list(dataloaders[0])[0].x[0].shape[0]})
        params["global_fdim"] = (
            first_batch.global_feats.shape[1]
            if hasattr(first_batch, "global_feats")
            else None
        )
        # print("params['in_channels']", params['in_channels'])
        # print("params['global_fdim']", params['global_fdim'])

        # gnn_model       = eval(f"{str(params['model'])}.from_dict({params})")
        # params['model'] = gnn_model
        # gnn_trainer     = GNNTrainer.from_dict(params)

        # print(f"params = {params}")
        # print("n_epochs", params['n_epochs'])
        # print("model", params['model'])
        # print('gnn_trainer', gnn_trainer.__dict__)
        # print('type', type(dataloaders[0]))

        if isinstance(dataloaders[0], (DataLoader, NodeLoader, LinkLoader)):
            val_score = self.train_and_validate(
                trial=trial,
                # gnn_trainer = gnn_trainer,
                dataloaders=dataloaders,
                params=params,
                split_mode=split_mode,
            )
            # print('val_score', val_score)

            return val_score

        else:
            raise TypeError(
                "train_val_data must be of type List[DataLoader] or , List[List[Data]]"
            )


    def run_optimization(
        self,
        # gnn_model:torch.nn.Module,
        train_val_data: Union[List[DataLoader], List[List[Data]]],
        params_grid: dict,
        # gnn_params_grid: dict,
        # trainer_params_grid:dict,
        optuna_direction: str,
        split_mode: str = "classic",
        study_name: str = None,
        **kwargs,
    ):
        ## Creating a study
        ## n_startup_trials = The number of initial trials that won’t be pruned. This helps build up a baseline of performance
        ## n_warmup_steps: The minimum number of reporting steps (epochs, batches, etc.) before a trial can be pruned. 
        # This ensures that each trial has a chance to reach some level of maturity.
        ## one can also consider 'interval_steps' to specify how often to check for pruning    
        self.study = create_study(
            direction=optuna_direction, sampler=self.sampler, study_name=study_name,
            pruner=MedianPruner(n_startup_trials=self.n_startup_trials ##
                                                , n_warmup_steps=self.n_warmup_steps)
        )

        objective = lambda trial: self.objective(
            # task=task,
            # gnn_model=gnn_model,
            trial=trial,
            train_val_data=train_val_data,
            params_grid=params_grid,
            # trainer_params_grid=trainer_params_grid,
            split_mode=split_mode,
            **kwargs,
        )

        objective.best_model = None
        objective.best_score = None

        self.study.optimize(
            objective, n_trials=self.n_trials, n_jobs=self.n_jobs, gc_after_trial=True
        )

        results = {
            "best_params": self.study.best_params,
            "best_score": self.study.best_value,
            # , "gnn_type": gnn_model.__class__.__name__
        }

        # Finalize TensorBoard logging
        if not self.tboard_writer is None:
            for trial in self.study.trials:
                status = "Complete" if trial.state == TrialState.COMPLETE else "Pruned"
                self.tboard_writer.add_text(f"Trial_{trial.number}/Final_Status", status)
                self.tboard_writer.add_hparams(trial.params, {"Final_Loss": trial.value or float("inf")})
            self.tboard_writer.close()


        return results


def pick_params(trial, params_grid: dict):
    params = {}
    for key, value in params_grid.items():
        # print(key, value, value.__class__, type(value), isinstance(value, type))

        if value is None:
            # print(key, value)
            params[key] = value

        elif type(value) == int:
            params[key] = trial.suggest_int(key, value, value)
            # print(key, params[key])
        elif type(value) == float:
            params[key] = trial.suggest_float(key, value, value)

        elif isinstance(value, str) or isinstance(value, bool):
            # params[key] = trial.suggest_categorical(key, value)
            params[key] = value

        elif isinstance(value, list) or isinstance(value, tuple):
            # print('\t=>type(value[0])', type(value[0]))
            if isinstance(value[0], str):
                params[key] = trial.suggest_categorical(key, value)
            elif isinstance(value[0], bool):
                params[key] = trial.suggest_categorical(key, value)
            elif (type(value[0]) in [int, float]) and (type(value[-1]) in [int, float]):
                if isinstance(value[0], int) and isinstance(value[-1], int):
                    if key in ["ffn_hidden_neurons", "gnn_hidden_neurons"]:
                        if len(value) == 2:
                            params[key] = trial.suggest_int(
                                key, min(value), max(value), step=16
                            )
                        elif len(value) > 2:
                            vals = [str(i) for i in value]
                            params[key] = int(trial.suggest_categorical(key, vals))
                        elif len(value) == 1:
                            params[key] = value[0]
                    elif key == "n_epochs":
                        params[key] = trial.suggest_int(
                            key, min(value), max(value), step=50
                        )
                    else:
                        params[key] = trial.suggest_int(key, min(value), max(value))
                elif isinstance(value[0], float) and isinstance(value[-1], float):
                    params[key] = trial.suggest_float(
                        key, min(value), max(value), log=(min(value) < 5e-3)
                    )

            elif (
                callable(value[0])
                or value[0].__class__.__name__ in ["MinMaxScaler", "StandarScaler"]
                or ".".join(str(type(value[0])).split(".")[:-1])
                in [
                    "torch.nn",
                    "torch.nn.modules.activation",
                    "torch.optim.lr_scheduler",
                    "sklearn.preprocessing",
                ]
            ):
                params[key] = value
            else:
                raise ValueError(f"Invalid values {value}.")

        elif (
            callable(value)
            or value.__class__.__name__ in ["MinMaxScaler", "StandarScaler"]
            or ".".join(type(value).split(".")[:-1])
            in [
                "torch.nn",
                "torch.nn.modules.activation",
                "torch.optim.lr_scheduler",
                "sklearn.preprocessing",
            ]
        ):
            ## or isinstance(value, torch.optim.lr_scheduler.LRScheduler) \
            ## or isinstance(value, torch.optim.Optimizer) \
            params[key] = value

        # elif isinstance(value, torch.optim.lr_scheduler.LRScheduler):
        #     params[key] = value

        elif type(value[0]) in [int, float]:
            if type(value[0]) == int:
                params[key] = trial.suggest_int(key, value, value)
            else:
                params[key] = trial.suggest_float(key, value, value)

        else:
            raise ValueError(f"Invalid values {value}.")
    # print("my params", params)
    return params


# def objective(trial):
#     # Suggest hyperparameters for the trial
#     gnn_hidden_neurons = trial.suggest_int("gnn_hidden_neurons", 64, 256, step=32)
#     gnn_nlayers = trial.suggest_int("gnn_nlayers", 2, 4)
#     ffn_hidden_neurons = trial.suggest_int("ffn_hidden_neurons", 64, 256, step=32)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

#     # Instantiate the model with suggested hyperparameters
#     model = GIN(
#         task="your_task",
#         in_channels=your_input_channels,
#         gnn_hidden_neurons=gnn_hidden_neurons,
#         gnn_nlayers=gnn_nlayers,
#         ffn_hidden_neurons=ffn_hidden_neurons,
#         dropout_rate=dropout_rate,
#         # ... other model parameters
#     )

#     # Optimizer with suggested learning rate
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     # Train the model
#     train(model, optimizer, train_loader, validate_loader, num_epochs=your_num_epochs)

#     # Return validation performance metric (e.g., accuracy, loss)
#     return validate_epoch(model, validate_loader)
