import re
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from lightning.pytorch.callbacks import Callback


class Tokenizer:
    def __init__(
        self,
        vocab: dict,
        max_len: int,
        no_processes: int,
        dataset_size: int,
        padding_token: str = "[PAD]",
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.dataset_size = dataset_size
        self.pad_token = padding_token
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
        # use padding
        token_list = tokens.split()
        if len(token_list) < self.max_len:
            token_list += [self.pad_token] * (self.max_len - len(token_list))
        if len(token_list) > self.max_len:
            print(f"Token length {len(token_list)} exceeds max length {self.max_len}")
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


class LogFigureCallback(Callback):
    #! code this up to log the r2 figure to wandb
    def plot_pred_vs_true(self, logits: np.ndarray, labels: np.ndarray, r2score: float):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hexbin(labels, logits, gridsize=100, cmap="viridis", bins="log")
        plt.colorbar()
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.set_title("Predicted vs True values")
        min_val, max_val = np.min([labels, logits]), np.max([labels, logits])
        ax.plot([min_val, max_val], [min_val, max_val],color='black')
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
    tokenizer = Tokenizer(vocab, 1000, no_cores, dataset_size)
    examples = examples["smi_can"].tolist()
    tokenized = tokenizer.tokenize(examples)
    max_len = max(len(token.split()) for token in tokenized)
    # get index for this max_len
    index = [i for i, token in enumerate(tokenized) if len(token.split()) == max_len]
    encoded = tokenizer.encode(tokenized)
    # get the length of first string in encoded
    print(encoded[index[0]])
