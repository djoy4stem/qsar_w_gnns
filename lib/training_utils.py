from os import getcwd, makedirs
from os.path import join, exists, isdir
from shutil import rmtree
from sklearn.metrics import r2_score, roc_auc_score, f1_score, mean_squared_error
import torch.optim as optim
from torch.optim import *
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import *
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, BCELoss, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn.utils import clip_grad_norm_
import torch
from typing import Optional, Any, Dict, Union
import numpy as np
from tqdm import tqdm
import warnings
import inspect
import ast

from typing import List


import sklearn
from sklearn import utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from lib import graph_nns, datasets, predictions
from lib.utilities import check_if_param_used

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
    ):
        self.task = task
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

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
            elif self.task == "binary_classification":
                self.criterion = BCELoss()
            elif self.task == "multilabel_classification":
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

    def train(self, train_loader, val_loader, save_every=0, n_epochs=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = 'cpu'
        self.model.to(device)

        print(f"device = {device}")
        print(f"TASK: {self.task}")

        val_scores, val_losses, train_losses = [], [], []
        if n_epochs is None:
            n_epochs = self.n_epochs
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(self.model, train_loader, device)
            train_losses.append(train_loss)

            val_loss, val_score = self.validate_epoch(self.model, val_loader, device)
            val_losses.append(val_loss)
            val_scores.append(val_score)

            if not self.scheduler is None:
                # self.scheduler.step()
                self.scheduler_step(val=val_loss)
                self.learning_rate = self.optimizer.param_groups[0]["lr"]

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
        losses = []
        batch_nr = 0
        for batch in train_loader:
            batch_nr += 1
            # print(f"Batch {batch_nr}")
            # for batch in tqdm(train_loader):
            batch = batch.to(device)
            output, train_true = predictions.predict_from_batch(
                batch=batch, model=model, return_true_targets=True
            )
            # print("\n", f"output[:5] = {output[:5]}\n")
            # print("\n", f"train_true[:5] = {train_true[:5]}\n")
            loss = None
            # print("\n", f"output[:5] = {output[:5]}\n") #, f"torch.max = {torch.max(output[:5], 1)}
            if self.task == "binary_classification":
                # print("train_pred", output.squeeze(1))
                loss = self.compute_loss(self.criterion, output.squeeze(1), train_true)

            elif self.task == "multilabel_classification":
                _, train_pred = torch.max(output, 1)
                loss = self.compute_loss(self.criterion, _, train_true)
                # print(f"train_pred = {train_pred.shape}", "\n", f"train_true = {train_true.view(-1,1).shape}")

            elif self.task == "regression":
                loss = self.compute_loss(self.criterion, output, train_true)
                # print(f"output = {output.shape}", "\n", f"train_true = {train_true.view(-1,1).shape}")

            else:
                raise ValueError(f"No implementation for task {self.task}.")

            if not loss is None:
                # print(f"train_true = {train_true.view(-1,1).shape}")
                # print(f"Criterion: {self.criterion}")
                # print(f"Train Loss = {loss}")
                losses.append(loss.item())
                loss.backward()
            else:
                print(
                    f"There was an issue computing the loss for training batch {batch_nr}. The loss was set to None, and will not be considered for averaging."
                )

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()  ## Update parameters based on gradients.
            self.optimizer.zero_grad()  ## Clear gradients.

        mean_loss = np.array(losses).mean()

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
            loss = None
            score = None

            if self.task in ["binary_classification", "multilabel_classification"]:
                if self.task == "binary_classification":
                    val_pred = val_pred.squeeze()
                    # print("val_pred", val_pred[:10])
                    loss = self.compute_loss(self.criterion, val_pred, val_true)
                else:
                    _, val_pred = torch.max(val_pred, 1)
                    loss = self.compute_loss(self.criterion, _, val_true)

                # print(f"val_pred({val_pred.shape}) = {val_pred[:5]}", "\n", f"val_true({val_true.shape}) = {val_true[:5]}")
                # print(f"Batch {batch_counter}: \n\t{val_pred.tolist()} \n\t{val_true.tolist()}")
                # print("torch.unique(val_true).size(0)", torch.unique(val_true).size(0))
                n_true_classes = torch.unique(val_true).size(0)

                # print(f"val_true ({utils.multiclass.type_of_target(val_true.cpu())}) = {val_true}")
                # # print(f"{utils.multiclass.type_of_target(val_true.cpu())}")
                # print(f"val_pred ({utils.multiclass.type_of_target(val_pred.cpu())}) = {val_pred}")
                # # print(f"{utils.multiclass.type_of_target(val_pred.cpu())}")

                if not loss is None:
                    if n_true_classes > 1:
                        # print(f"val_pred = {val_pred}")
                        if self.scoring_func.__name__ in ["roc_auc_score"]:
                            score = self.scoring_func(
                                val_true.cpu(), val_pred.detach().cpu()
                            )
                            scores.append(score)
                        elif self.scoring_func.__name__ in [
                            "balanced_accuracy_score",
                            "precision_score",
                            "recall_score",
                        ]:
                            ## Classification metrics can't handle a mix of binary and continuous targets
                            val_pred_classes = None
                            if self.task == "binary_classification":
                                val_pred_classes = [
                                    int(p > threshold) for p in val_pred.detach().cpu()
                                ]
                            else:
                                val_pred_classes = [
                                    v.index(max(v)) for c in val_pred.detach().cpu()
                                ]

                            # print('val_pred_classes', val_pred_classes[:10])
                            # print('val_pred_true', val_true.cpu()[:10])
                            score = self.scoring_func(val_true.cpu(), val_pred_classes)
                        # print(f"\tBatch counter {batch_counter}: score = {score}")

                    elif n_true_classes == 1:
                        if self.scoring_func.__name__ == "roc_auc_score":
                            warnings.warn(
                                "Only one class present in y_true. ROC AUC score is not defined in that case. This bastch will be skipped."
                            )
                        elif self.scoring_func.__name__ != "roc_auc_score":
                            warnings.warn(
                                "Caution. There is only one class, which means the set is missing true positives or true negatives, potentially resulting in values of zero."
                            )

            elif self.task == "regression":
                loss = self.compute_loss(self.criterion, val_pred, val_true)
                score = self.scoring_func(val_true.cpu(), val_pred.detach().cpu())
                scores.append(score)
                # print(f"output = {output.shape}", "\n", f"val_true = {val_true.view(-1,1).shape}")

            else:
                raise ValueError(f"No implementation for task {self.task}.")

            # print(f"train_true = {train_true.view(-1,1).shape}")
            # print(f"Criterion: {self.criterion}")
            # print(f"Val Loss = {loss}")

            if not loss is None:
                losses.append(loss.item())
                loss.backward()
            else:
                print(
                    f"There was an issue computing the loss for training batch {batch_counter}. The loss was set to None, and will not be considered for averaging."
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

    # def train_epoch(self, model, train_loader, device):
    #         model.train()
    #         train_pred, train_true = predictions.predict_from_loader(train_loader, model=model
    #                                                                , task=self.task, return_true_targets=True
    #                                                                , device=device
    #                                                                , desc="Predicting train targets...")
    #         # print(f"train_pred = {train_pred.shape}", "\n", f"train_true = {train_true.view(-1,1).shape}")
    #         # print(f"train_true = {train_true.view(-1,1).shape}")
    #         # print(f"Criterion: {self.criterion}")
    #         loss = self.compute_loss(self.criterion, train_pred, train_true)
    #         # if self.task == 'regression':
    #         #     loss = self.criterion(train_pred, train_true.view(-1,1))
    #         #     # loss.backward()
    #         # else:
    #         #     loss = self.criterion(train_pred, train_true)
    #             # loss.backward()
    #         # loss = self.criterion(train_pred, train_true)
    #         # loss.backward()
    #         # print(f"Loss: {loss}")
    #         loss.backward()

    #         self.optimizer.step() ## Update parameters based on gradients.
    #         self.optimizer.zero_grad() ## Clear gradients.
    #         if not self.scheduler is None:
    #             self.scheduler.step

    #         return loss.item()

    # def validate_epoch(self, model, val_loader, device):
    #     model.eval()
    #     val_pred, val_true = predictions.predict_from_loader(val_loader, model=model, task=self.task
    #                                                        , return_true_targets=True, device=device
    #                                                        , desc="Predicting validation targets...")
    #     # loss = None
    #     # if self.task == 'regression':
    #     #     loss = self.criterion(val_pred, val_true.view(-1,1))
    #     # else:
    #     #     loss = self.criterion(val_pred, val_true)
    #     loss = self.compute_loss(self.criterion, val_pred, val_true)
    #     loss.backward()
    #     # print('self.scoring_func', self.scoring_func)
    #     score = self.scoring_func(val_pred.detach().cpu(), val_true.cpu())

    #     return loss.item(), score

    def compute_loss(self, criterion, pred_target, true_target):
        # if True:
        try:
            if self.task == "regression":
                return self.criterion(pred_target, true_target.view(-1, 1))

            elif self.task == "binary_classification":
                # print(pred_target.shape, true_target.shape)
                # print([x for x in pred_target.float() if x <0 or x > 1])
                # print([x for x in true_target if x <0 or x > 1])
                # print("pred_target:", pred_target.float())
                # print(true_target)
                return self.criterion(pred_target.float(), true_target)

            elif self.task == "multilabel_classification":
                return self.criterion(pred_target, true_target)
            else:
                raise ValueError("Task not supported for loss computation.")
        except Exception as exp:
            return None


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


import optuna
from sklearn.metrics._scorer import _PredictScorer


class OptunaHPO:
    def __init__(self, n_trials=5, n_jobs=-1, sampler=None):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler

    def get_params(self):
        return {
            "n_trials": self.n_trials,
            "n_jobs": self.n_jobs,
            "sampler": self.sampler,
        }

    def train_and_validate(
        self,
        # gnn_trainer: GNNTrainer,
        params: dict,
        dataloaders: List[DataLoader],
        split_mode: str = "classic",
    ):
        if split_mode == "classic":
            
            gnn_params = {p:params[p] for p in params if check_if_param_used(eval(str(params['model'])), p) }
           
            gnn_model = eval(f"{str(params['model'])}.from_dict({gnn_params})")

            # task = params['task']
            # params = gnn_params
            # params['task'] = task
            params["model"] = gnn_model
            gnn_trainer = GNNTrainer.from_dict(params)

            train_losses, val_losses, val_scores = gnn_trainer.train(
                train_loader=dataloaders[0],
                val_loader=dataloaders[1],
                save_every=0,
                n_epochs=None,
            )

            return val_scores[-1]
            # return val_losses[-1]
        else:
            raise NotImplementedError("train_validate is only implemented for the 'classic' split mode.")

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
        # print(f"0 params = {params}")

        dataloaders = train_val_data

        if not isinstance(train_val_data[0], DataLoader) and isinstance(
            train_val_data[0][0], Data
        ):
            dl_params = {}
            batch_size = params.get("batch_size", 128)
            shuffle = params.get("shuffle", False)
            add_global_feats_to_nodes = params.get("add_global_feats_to_nodes", False)
            num_workers = params.get("num_workers", 0)
            standardize = params.get("standardize", True)
            standardizer = params.get("standardizer", MinMaxScaler())

            print("Creating DataLoader objects...")
            train_loader = datasets.get_dataloader(
                dataset=train_val_data[0],
                batch_size=batch_size,
                shuffle=shuffle,
                add_global_feats_to_nodes=add_global_feats_to_nodes,
                num_workers=num_workers,
                standardize=standardize,
                standardizer=standardizer,
            )
            val_loader = datasets.get_dataloader(
                dataset=train_val_data[1],
                batch_size=batch_size,
                shuffle=shuffle,
                add_global_feats_to_nodes=add_global_feats_to_nodes,
                num_workers=num_workers,
                standardize=standardize,
                standardizer=standardizer,
            )
            dataloaders = [train_loader, val_loader]

        params["in_channels"] = list(dataloaders[0])[0].x[0].shape[0]
        # params.update(**{'in_channels' : list(dataloaders[0])[0].x[0].shape[0]})
        params["global_fdim"] = (
            list(dataloaders[0])[0].global_feats.shape[1]
            if "global_feats" in list(dataloaders[0])[0].to_dict()
            else None
        )
        # print("params['in_channels']", params['in_channels'])
        # print("params['global_fdim']", params['global_fdim'])

        # gnn_model       = eval(f"{str(params['model'])}.from_dict({params})")
        # params['model'] = gnn_model
        # gnn_trainer     = GNNTrainer.from_dict(params)

        print(f"params = {params}")
        # print("n_epochs", params['n_epochs'])
        # print("model", params['model'])
        # print('gnn_trainer', gnn_trainer.__dict__)
        # print('type', type(dataloaders[0]))

        if isinstance(dataloaders[0], DataLoader):
            val_score = self.train_and_validate(
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
        self.study = optuna.create_study(
            direction=optuna_direction, sampler=self.sampler, study_name=study_name
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

        self.study.optimize(
            objective, n_trials=self.n_trials, n_jobs=self.n_jobs, gc_after_trial=True
        )

        results = {
            "best_params": self.study.best_params,
            "best_score": self.study.best_value
            # , "gnn_type": gnn_model.__class__.__name__
        }

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
