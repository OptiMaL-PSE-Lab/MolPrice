import os
import timeit
from typing import Union

import gin
import torch
import numpy as np
import lightning.pytorch as L
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore

from src.plotter import plot_parity, plot_dist_overlap
from src.data_loader import TestLoader
from src.path_lib import *
from src.model_utils import load_checkpointed_gin_config, load_model_from_checkpoint


def main_data_test(
    args,
    model: LightningModule,
    loader: TestLoader,
):

    loaded_model = load_model_from_checkpoint(model, CHECKPOINT_PATH / args.cn)
    trainer = L.Trainer(
        accelerator="auto", devices=find_usable_cuda_devices(), logger=False
    )

    out = trainer.test(loaded_model, loader.test_dataloader())

    # test inference speed on test set (one prediction at a time i.e. batch size 1)
    loader.batch_size = 1  # type: ignore
    my_loader = loader.test_dataloader()
    time_start = timeit.default_timer()
    trainer.predict(loaded_model, my_loader, return_predictions=False)
    end_time = timeit.default_timer()

    avg_time = (end_time - time_start) / len(my_loader.dataset)  # type: ignore
    # write out to CONFIG_PATH with new line between each metric
    with open(CONFIG_PATH / "test_results.txt", "w") as f:
        for metric in out:
            f.write(f"{metric}\n")  #
        f.write(f"Average inference time: {avg_time}\n")
    print(out)

    if args.plot:
        labels, logits = loaded_model.test_predictions, loaded_model.test_labels
        r2_score = out[-1]["r2_score"]
        labels_np = np.concatenate([label.cpu().numpy() for label in labels], axis=0)  # type: ignore
        logits_np = np.concatenate([logit.cpu().numpy() for logit in logits], axis=0)  # type: ignore
        fig = plot_parity(logits_np, labels_np, r2_score)
        fig.savefig(CONFIG_PATH / "parity.png", dpi=300)


def main_ood_test(
    args,
    model: LightningModule,
    loader: Union[LightningDataModule, list[LightningDataModule]],
):

    loaded_model = load_model_from_checkpoint(model, CHECKPOINT_PATH / args.cn)
    trainer = L.Trainer(
        accelerator="auto", devices=find_usable_cuda_devices(), logger=False
    )
    if isinstance(loader, list):
        test_name = args.test_name.split(",")
        outputs = {}
        for i, load in enumerate(loader):
            t_name = test_name[i].split("/")[-1]
            out = trainer.predict(loaded_model, load.test_dataloader())
            outputs[f"{t_name}"] = torch.cat(out).numpy()  # type:ignore
    else:
        t_name = args.test_name.split("/")[-1]
        out = trainer.predict(loaded_model, loader.test_dataloader())
        outputs[f"{t_name}"] = torch.cat(out).numpy()  # type: ignore

    if args.plot:
        fig = plot_dist_overlap(outputs)
        fig.savefig(CONFIG_PATH / "dist_overlap.png", dpi=300)


def add_shared_arguments(parser):
    """Function to add shared arguments to a parser."""
    parser.add_argument(
        "--model",
        type=str,
        help="Model to test",
        required=True,
        choices=["LSTM_EFG", "LSTM_IFG", "Transformer", "Fingerprint"],
    )

    parser.add_argument(
        "--checkpoint_name",
        "--cn",
        dest="cn",
        type=str,
        help="Name of the checkpoint file to load",
        required=True,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the parity plot",
    )

    parser.add_argument(
        "--combined",
        action="store_true",
        help="If combined model has been used, activate --combined flag"
    )


if __name__ == "__main__":
    import argparse

    from src.model import FgLSTM, TransformerEncoder, Fingerprints
    from src.data_loader import TestLoader

    model_dict = {
        "FgLSTM": FgLSTM,
        "Transformer": TransformerEncoder,
        "Fingerprint": Fingerprints,
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    data_parser = subparsers.add_parser("main_data")
    add_shared_arguments(data_parser)
    data_parser.add_argument("--has_price", action="store_false")
    data_parser.add_argument(
        "--test_name",
        type=str,
        help="Name of the test file within ./testing",
        required=True,
    )
    data_parser.set_defaults(func=main_data_test)

    ood_parser = subparsers.add_parser("main_ood")
    add_shared_arguments(ood_parser)
    ood_parser.add_argument("--has_price", action="store_true")
    ood_parser.add_argument(
        "--test_name",
        type=str,
        help="Name of the test file(s) within ./testing, separated by comma",
        required=True,
    )
    ood_parser.set_defaults(func=main_ood_test)

    args = parser.parse_args()

    model = model_dict[args.model]

    test_paths = args.test_name.split(",")
    if len(test_paths) > 1:
        test_loader = [
            TestLoader(
                TEST_PATH / test,
                DATA_PATH,
                128,
                model_name=args.model,
                has_price=args.has_price,
            )
            for test in test_paths
        ]
    else:
        test_loader = TestLoader(
            TEST_PATH / args.test_name,
            DATA_PATH,
            128,
            model_name=args.model,
            has_price=args.has_price,
        )

    # read gin file
    CONFIG_PATH = CHECKPOINT_PATH.joinpath(args.cn).parent

    # read config file as txt, delete first line, save as temporary file, parse file, delete temporary file
    load_checkpointed_gin_config(CONFIG_PATH, caller="test")

    args.func(args, model, test_loader)
