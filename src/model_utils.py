import gin
import math
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gin import query_parameter as gin_qp
from tqdm import tqdm
from multiprocessing import Pool
from lightning.pytorch.callbacks import Callback


class Tokenizer:
    def __init__(
        self,
        vocab: dict,
        no_processes: int,
        dataset_size: int,
    ):
        self.vocab = vocab
        self.dataset_size = dataset_size
        self.no_processes = no_processes

    def tokenize(self, smi: str | list[str]) -> str | list[str]:
        if isinstance(smi, list):
            return self._batch_tokenize(smi)
        elif isinstance(smi, str):
            return self._tokenize(smi)

    def encode(self, tokens: str | list[str]) -> list[int] | list[list[int]]:
        if isinstance(tokens, list):
            return self._batch_encode(tokens)
        elif isinstance(tokens, str):
            return self._encode(tokens)

    def _tokenize(self, smi: str) -> str:
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"  # type:ignore
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == "".join(tokens)
        tokens.insert(0, "[CLS]")
        return " ".join(tokens)

    def _batch_tokenize(self, smiles: list[str]) -> list[str]:
        with Pool(processes=self.no_processes) as pool:
            tokenized = list(
                tqdm(
                    pool.imap(self.tokenize, smiles, chunksize=2 * self.no_processes),
                    total=self.dataset_size,
                )
            )
        return tokenized  # type:ignore

    def _encode(self, tokens: str) -> list[int]:
        token_list = tokens.split()
        # encode tokens
        return [self.vocab[token] for token in token_list]

    def _batch_encode(self, tokens: list[str]) -> list[list[int]]:
        with Pool(processes=self.no_processes) as pool:
            encoded = list(
                tqdm(
                    pool.imap(self._encode, tokens, chunksize=2 * self.no_processes),
                    total=self.dataset_size,
                )
            )
        return encoded
    

def calculate_max_training_step(path_data) -> None:
    batch_size, acc_batches, epochs = gin_qp("%batch_size"), gin_qp("main.gradient_accum"), gin_qp("main.max_epoch")
    dataframe_name, splits = gin_qp("%df_name"), gin_qp("%data_split")
    df = pd.read_csv(path_data / dataframe_name)
    len_train = df.shape[0] * splits[0]
    batches_per_gpu = math.ceil(len_train / float(batch_size))
    train_steps = math.ceil(batches_per_gpu / acc_batches) * epochs
    gin.bind_parameter("transformer/torch.optim.lr_scheduler.OneCycleLR.total_steps", train_steps) 

class LogFigureCallback(Callback):
    #! code this up to log the r2 figure to wandb
    def plot_pred_vs_true(self, logits: np.ndarray, labels: np.ndarray, r2score: float):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hexbin(labels, logits, gridsize=100, cmap="viridis", bins="log")
        plt.colorbar()
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.set_title("Predicted vs True values")
        min_val, max_val = np.min([labels, logits]), np.max([labels, logits])
        ax.plot([min_val, max_val], [min_val, max_val], color="black")
        ax.legend(["R2 score: {:.3f}".format(r2score)])
        return fig


if __name__ == "__main__":
    import os
    import pandas as pd
    from pathlib import Path

    file_path = Path(__file__).parent.parent
    vocab_path = file_path / "data/vocab" / "vocab_SMILES.txt"
    smiles_path = file_path / "data" / "chemspace_reduced.csv"
    with open(vocab_path, "r") as f:
        vocab = {line.strip(): idx for idx, line in enumerate(f)}

    no_cores = os.cpu_count() - 2  # type: ignore
    examples = pd.read_csv(smiles_path, header=0)
    dataset_size = examples.shape[0]
    tokenizer = Tokenizer(vocab, no_cores, dataset_size)
    examples = examples["smi_can"].tolist()
    tokenized = tokenizer.tokenize(examples)
    max_len = max(len(token.split()) for token in tokenized)
    # get index for this max_len
    index = [i for i, token in enumerate(tokenized) if len(token.split()) == max_len]
    encoded = tokenizer.encode(tokenized)
    # get the length of first string in encoded
    print(encoded[index[0]])
