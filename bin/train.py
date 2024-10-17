import os
import gin
import pytorch_lightning as L
import datetime
from datetime import date
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning import LightningModule

from src.data_loader import CustomDataLoader
import gin.torch.external_configurables

# For running on cluster - slow connection to wandb
os.environ["WANDB_INIT_TIMEOUT"] = "1000"
os.environ["WANDB_HTTP_TIMEOUT"] = "1000"


@gin.configurable(module="bin.train")
def main(
    args,
    model: LightningModule,
    data_module: CustomDataLoader,
    max_epoch: int,
    early_stopping: bool,
    patience: int,
    no_gpus: int,
    logging: bool,
    gradient_accum: int,
):

    today = date.today()
    hour = datetime.datetime.now().hour
    d = today.strftime("%m%d%Y")
    checkpoint_path = args.checkpoint_path / f"{args.model}-{args.fp}-{d}-{hour}"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        dirpath=checkpoint_path,
        filename="{epoch}-{val_loss:.3f}",
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    if early_stopping:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=True,
            patience=patience,
        )
    else:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss", mode="min", patience=max_epoch
        )

    callbacks = [checkpoint_callback, lr_callback, earlystopping_callback]

    # initialize logger
    if logging:
        logger = WandbLogger(
            project="price_molport",
            name=f"{args.model}-{args.fp}-{d}-{hour}",
            log_model="all",
            save_dir=args.log_path,
        )
        logger.watch(model)
    else:
        logger = CSVLogger(save_dir=args.log_path)

    if no_gpus > 0:
        torch.set_float32_matmul_precision("medium")
        devices = [i for i in range(no_gpus)]
        trainer = L.Trainer(
            accelerator="cuda",
            devices=devices,
            max_epochs=max_epoch - 1,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            log_every_n_steps=500,
            accumulate_grad_batches=gradient_accum,
            enable_progress_bar=True,
        )
    else:
        print("Training resumes on CPU.")
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=max_epoch - 1,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            log_every_n_steps=500,
            enable_progress_bar=False,
        )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    # within the checkpoint path, put info about the loggers files and the gin config
    config_info = gin.operative_config_str()
    loggers_id = logger.version if logging else "version_" + str(logger.version)  # type: ignore
    # write to new file the config_info and loggers_id
    with open(checkpoint_path / "config_info.txt", "w") as f:
        f.write(f"Loggers ID: {loggers_id}")
        f.write("\n")
        f.write(config_info)


if __name__ == "__main__":
    import gin
    import torch
    from argparse import ArgumentParser

    from src.model import FgLSTM, TransformerEncoder, Fingerprints
    from src.model_utils import (
        calculate_max_training_step,
        load_checkpointed_gin_config,
    )
    from src.data_loader import EFGLoader, IFGLoader, FPLoader, TFLoader, CombinedLoader
    from src.path_lib import *

    loader_dict = {
        "LSTM_EFG": EFGLoader,
        "LSTM_IFG": IFGLoader,
        "Fingerprint": FPLoader,
        "Transformer": TFLoader,
    }

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to train",
        required=True,
        choices=["LSTM_EFG", "LSTM_IFG", "Transformer", "Fingerprint"],
    )
    parser.add_argument(
        "--fingerprint_type",
        "--fp",
        dest="fp",
        type=str,
        help="Type of fingerprint to use",
        required=False,
        choices=["morgan", "rdkit", "atom", "mhfp"],
        default="morgan",
    )

    parser.add_argument(
        "--checkpoint_name",
        "--cn",
        dest="cn",
        type=str,
        help="Name of the checkpoint file to load",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--combined",
        action="store_true",
        help="Whether to use combined dataloaders to train via contrastive learning",
    )

    # Parsing args
    args = parser.parse_args()
    args.checkpoint_path = CHECKPOINT_PATH
    args.log_path = path / "logs"

    # Parsing gin configs depending on if checkpoint is used
    if args.cn:
        gin.constant("loss_sep", True)
        gin_checkpoint = (CHECKPOINT_PATH / args.cn).parent
        load_checkpointed_gin_config(gin_checkpoint, "train")
    else:
        gin.constant("loss_sep", False)
        gin.parse_config_file(GIN_PATH_MODEL)
        gin.bind_parameter("FPLoader.fp_type", args.fp)

    # Querying current dataset used for training
    current_dataset = gin.query_parameter("%df_name").split(".")[0]
    feature_path = DATA_PATH / "features" / current_dataset

    # Initialising models from checkpoint or from scratch
    model_name = args.model.split("_")[0]
    model_dict = {
        "LSTM": FgLSTM,
        "Transformer": TransformerEncoder,
        "Fingerprint": Fingerprints,
    }

    if model_name == "Transformer":
        calculate_max_training_step(
            DATABASE_PATH
        )  # * Specific to the scheduler used (i.e. OneCycleLR)

    if isinstance(args.cn, str):
        # * Load weights of model, hyperparameters from checkpoint config.txt (to allow for changes in hyperparameters)
        ckpt_dict = torch.load(CHECKPOINT_PATH / args.cn)
        state_dict = ckpt_dict["state_dict"]
        model = model_dict[model_name](gin.REQUIRED)
        model.load_state_dict(state_dict, strict=False)
        if args.combined and args.model == "Fingerprint":
            # * Freeze the last layer weights to learn proper latent space representation
            model.linear.weight.requires_grad = False
        print("Model loaded - training resumes.")
    else:
        model = model_dict[model_name](
            gin.REQUIRED
        )  # loaded hps from model_configs.gin

    # Dataloader and Model initialization depending on combined behaviour
    data_object = loader_dict[args.model]
    if args.combined:
        print("Using combined dataload. Only implemented for FP models for now.")
        # * for now default dataset given by GASA for HS
        feature_path = DATA_PATH / "features" / "molport_reduced"
        feature_path = DATA_PATH / "features" / "molport_reduced"
        es_dataloader = data_object(
            data_path=DATABASE_PATH, feature_path=feature_path, hp_tuning=False
        )
        hs_path = TEST_PATH / "gasa"
        hs_dataloader = data_object(
            data_path=hs_path,
            feature_path=hs_path,
            hp_tuning=False,
            df_name="test_hs.csv",
        )
        data_module = CombinedLoader(es_dataloader, hs_dataloader)
    else:
        data_object = loader_dict[args.model]
        data_module = data_object(
            data_path=DATABASE_PATH, feature_path=feature_path, hp_tuning=False
        )

    gin.finalize()  # not allowing any more changes to the config
    main(args, model=model, data_module=data_module, max_epoch=gin.REQUIRED, early_stopping=gin.REQUIRED, patience=gin.REQUIRED, no_gpus=gin.REQUIRED, logging=gin.REQUIRED)  # type: ignore
