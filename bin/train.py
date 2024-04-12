import gin
import wandb
import lightning as L
from datetime import date
from pathlib import Path
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch import LightningModule, LightningDataModule

import gin.torch.external_configurables


@gin.configurable
def main(
    args,
    model: LightningModule,
    data_module: LightningDataModule,
    max_epoch: int,
    early_stopping: bool,
    patience: int,
    no_gpus: int,
    logging: bool,
):

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        dirpath=args.checkpoint_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    if early_stopping:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=0.01
        )
    else:
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=max_epoch)

    model_summary_callback = ModelSummary(max_depth=1)

    callbacks = [checkpoint_callback, earlystopping_callback, model_summary_callback]

    # initialize logger
    if logging:
        wandb.login()
        today = date.today()
        d = today.strftime("%m/%d/%Y")
        logger = WandbLogger(
            project="graphfg", name=f"model-{args.model}-{d}", log_model="all", save_dir=args.log_path 
        )
        logger.watch(model)
    else:
        logger = True

    if no_gpus>0:
        devices = find_usable_cuda_devices()
        trainer = L.Trainer(
            accelerator="cuda",
            devices=devices,
            max_epochs=max_epoch,
            logger=logger,
            callbacks=callbacks,
            limit_train_batches=0.001,
            limit_val_batches=0.001,
        )
    else:
        print("Training resumes on CPU.")
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=max_epoch,
            logger=logger,
            callbacks=callbacks,
            limit_train_batches=0.001,
            limit_val_batches=0.001,
        )

    trainer.fit(
        model,
        datamodule=data_module,
    )


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
    checkpoint_path = path / "models" / "temp"
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
        default = "Transformer"
    )
    parser.add_argument(
        "--fingerprint_type",
        "--fp",
        dest="fp",
        type=str,
        help="Type of fingerprint to use",
        required=False,
        choices=["morgan", "rdkit", "atom"],
        default = "morgan"
    )

    args = parser.parse_args()
    args.checkpoint_path = checkpoint_path
    args.log_path = path / "logs"
    gin.parse_config_file(gin_path_dataloader)
    gin.bind_parameter('FPLoader.fp_type', args.fp)  

    data_object = loader_dict[args.model]
    data_module = data_object(
        data_path=data_path, feature_path=feature_path
    )  

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
