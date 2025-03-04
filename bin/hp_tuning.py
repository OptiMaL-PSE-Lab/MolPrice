"""
After determining best model architecture, this file is used to tune Fingerprint hyperparameters
"""

import os

import gin
import optuna
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback  # type: ignore

from src.model import Fingerprints
from src.data_loader import FPLoader


def objective_single_price(trial: optuna.trial.Trial) -> float:
    # check if there is an activate wandb session
    if wandb.run:
        wandb.log({"pruned": 1})
        wandb.finish()
    # optimize learning rate, batch size, fingerprint size, hidden_dim, dropout
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.05)
    fingerprint_size = trial.suggest_categorical("fingeprint_size", [2048, 4096])
    hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [512, 1024])
    hidden_dim_2 = trial.suggest_categorical("hidden_dim_2", [256, 512])
    hidden_dim_3 = trial.suggest_categorical("hidden_dim_3", [64, 128, 256])
    two_d = trial.suggest_categorical("two_d", [True, False])
    count_simulation = trial.suggest_categorical("count_simulation", [True, False])
    fp_type = trial.suggest_categorical("fp_type", ["morgan", "rdkit", "mhfp", "atom"])

    current_dataset = gin.query_parameter("%df_name").split(".")[0]
    feature_path = DATA_PATH / "features" / f"{current_dataset}/hp_tuning"
    # create feature path if it does not exist
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    config = dict(trial.params)
    config["trial_number"] = trial.number

    # set up callbacks
    save_path = path / "logs" / STUDY_NAME
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = WandbLogger(
        project=STUDY_NAME,
        config=config,
        save_dir=save_path,
        log_model=False,
        reinit=True,
    )

    # set up the data loader
    data_object = FPLoader(
        data_path=DATABASE_PATH,
        feature_path=feature_path,
        batch_size=batch_size,
        count_simulation=count_simulation,
        fp_type=fp_type,
        two_d=two_d,
        workers_loader=gin.REQUIRED,
        data_split=gin.REQUIRED,
        df_name=gin.REQUIRED,
        fp_size=fingerprint_size,
        p_r_size=gin.REQUIRED,  # type: ignore
        hp_tuning=True,
    )

    # set up the model
    model = Fingerprints(
        input_size=fingerprint_size,
        hidden_size_1=hidden_dim_1,
        hidden_size_2=hidden_dim_2,
        hidden_size_3=hidden_dim_3,
        dropout=dropout,
        two_d=two_d,
        loss_sep=False,
        loss_hp=0,
    )
    gin.parse_config_file(GIN_PATH_TUNING)
    gin.bind_parameter("torch.optim.Adam.lr", lr)
    gin.bind_parameter("torch.optim.Adam.weight_decay", weight_decay)

    trainer = Trainer(
        logger=logger,
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        enable_checkpointing=False,
        log_every_n_steps=1000,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=data_object)
    # get the best validation loss from the model
    val_summary = wandb.run.summary if wandb.run else None
    if val_summary:
        val_loss = val_summary["val_loss"]
    else:
        val_loss = 0.0

    # log the final validation loss to wandb logger
    logger.log_metrics({"pruned": 0})
    wandb.finish()

    return val_loss  # type: ignore


if __name__ == "__main__":
    from src.path_lib import *

    gin.parse_config_file(GIN_PATH_DATALOADER)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    STUDY_NAME = "hp_tuning"
    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name=STUDY_NAME
    )
    study.optimize(objective_single_price, n_trials=35)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # plot optimization history with wandb
    wandb.init(project=STUDY_NAME)

    wandb.log(
        {
            "optuna_optimization_history": optuna.visualization.plot_optimization_history(
                study
            ),
            "optuna_param_importance": optuna.visualization.plot_param_importances(
                study
            ),
        }
    )
