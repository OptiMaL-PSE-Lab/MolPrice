import os
import gin
import lightning as L
import datetime
from datetime import date
from pathlib import Path
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch import LightningModule
from lightning.pytorch.tuner import Tuner  # type: ignore

from src.data_loader import CustomDataLoader
import gin.torch.external_configurables

# For running on cluster - slow connection to wandb
os.environ["WANDB_INIT_TIMEOUT"] = "1000"
os.environ["WANDB_HTTP_TIMEOUT"] = "1000"


@gin.configurable
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
    d = today.strftime("%m/%d/%Y")
    checkpoint_path = args.checkpoint_path / f"{args.model}-{args.fp}-{d}-{hour}"
    os.mkdir(checkpoint_path)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        dirpath=checkpoint_path,
        filename="{epoch}-{val_loss:.3f}",
    )

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

    callbacks = [checkpoint_callback, earlystopping_callback]

    # initialize logger
    if logging:
        logger = WandbLogger(
            project="graphfg",
            name=f"{args.model}-{args.fp}-{d}-{hour}",
            id = f"{args.model}-{args.fp}-{d}-{hour}",
            log_model="all",
            save_dir=args.log_path,
        )
        logger.watch(model)
    else:
        logger = CSVLogger(save_dir=args.log_path)

    if no_gpus > 0:
        devices = find_usable_cuda_devices()
        trainer = L.Trainer(
            accelerator="cuda",
            devices=devices,
            max_epochs=max_epoch,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            log_every_n_steps=500,
        )
    else:
        print("Training resumes on CPU.")
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=max_epoch,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            log_every_n_steps=500,
        )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data_module, mode="power", max_trials=7)

    trainer.fit(
        model,
        datamodule=data_module,
    )

    # within the checkpoint path, put info about the loggers files and the gin config 
    config_info = gin.operative_config_str()
    loggers_id = logger._id if logging else logger.version # type: ignore
    # write to new file the config_info and loggers_id 
    with open(checkpoint_path / "config_info.txt", "w") as f:
        f.write(f"Loggers ID: {loggers_id}")
        f.write("\n")
        f.write(config_info)


if __name__ == "__main__":
    import gin
    from pathlib import Path
    from argparse import ArgumentParser

    from src.model import FgLSTM, TransformerEncoder, Fingerprints
    from src.data_loader import EFGLoader, IFGLoader, FPLoader, TFLoader

    # Set up all paths
    path = Path(__file__).parent.parent
    gin_path_model = str(path / "configs" / "model_configs.gin")
    gin_path_dataloader = str(path / "configs" / "dataloader.gin")
    data_path = path / "data"
    feature_path = data_path / "features"
    checkpoint_path = path / "models"
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
        required=False,
        choices=["LSTM_EFG", "LSTM_IFG", "Transformer", "Fingerprint"],
    )
    parser.add_argument(
        "--fingerprint_type",
        "--fp",
        dest="fp",
        type=str,
        help="Type of fingerprint to use",
        required=False,
        choices=["morgan", "rdkit", "atom"],
        default="morgan",
    )

    args = parser.parse_args()
    args.checkpoint_path = checkpoint_path
    args.log_path = path / "logs"
    gin.parse_config_file(gin_path_dataloader)
    gin.bind_parameter("FPLoader.fp_type", args.fp)

    data_object = loader_dict[args.model]
    data_module = data_object(data_path=data_path, feature_path=feature_path)

    # parse model gin file after data_object has been loaded
    gin.parse_config_file(gin_path_model)
    gin.finalize()
    model_name = args.model.split("_")[0]
    model_dict = {
        "LSTM": FgLSTM,
        "Transformer": TransformerEncoder,
        "Fingerprint": Fingerprints,
    }
    model = model_dict[model_name](gin.REQUIRED)
    main(args, model=model, data_module=data_module, max_epoch=gin.REQUIRED, early_stopping=gin.REQUIRED, patience=gin.REQUIRED, no_gpus=gin.REQUIRED, logging=gin.REQUIRED)  # type: ignore
