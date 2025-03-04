import lightning.pytorch as L
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore

from src.data_loader import TestLoader
from src.path_lib import *
from src.model_utils import load_checkpointed_gin_config, load_model_from_checkpoint


def main(args, model, loader: TestLoader):
    """
    Inference function for the model
    """
    loaded_model = load_model_from_checkpoint(model, CHECKPOINT_PATH / args.cn)
    trainer = L.Trainer(
        accelerator="auto",
        devices=find_usable_cuda_devices(),
        logger=False,
    )
    out = trainer.predict(loaded_model, loader.test_dataloader())
    del loaded_model, trainer
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import torch
    from src.model import Fingerprints

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--mol",
        help="Path to the molecule file (.csv) or singular SMILES string",
        required=True,
    )
    parser.add_argument("--cn", help="Model Checkpoint", required=True)
    args = parser.parse_args()
    args.cn = f"{args.cn}/best.ckpt"

    CONFIG_PATH = CHECKPOINT_PATH.joinpath(args.cn).parent
    load_checkpointed_gin_config(CONFIG_PATH, caller="test", combined=False)

    test_loader = TestLoader(
        args.mol,
        DATABASE_PATH,
        batch_size=128,
        model_name="Fingerprint",
        has_price=False,
    )

    out = main(args, Fingerprints, test_loader)
    out = [x[0] for x in out]  # type: ignore

    if ".csv" in args.mol:
        out = torch.cat(out).squeeze().tolist()
        df = pd.read_csv(args.mol)
        smi = df["smi_can"].to_list()
        out_dict = {"smi_can": smi, "score": out}
        out_df = pd.DataFrame(out_dict).round(decimals=2)
        out_df.to_csv(f"prices.csv", index=False)
    else:
        print(f"The predicted price for {args.mol} is {float(out[0]).__round__(2)}!")
