import gin
import math
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gin import query_parameter as gin_qp
from tqdm import tqdm
from typing import Optional
from multiprocessing import Pool
from lightning.pytorch.callbacks import Callback


class Tokenizer:
    def __init__(
        self,
        no_processes: int,
        dataset_size: int,
        vocab: Optional[dict[str, int]] = None,
    ):
        self.dataset_size = dataset_size
        self.no_processes = no_processes
        self.vocab = vocab

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
        
    def build_vocab(self, tokens: str | list[str]) -> dict[str, int]:
        print("Building vocabulary...")
        # add special tokens 
        vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        for token in tqdm(tokens):
            token = token.split(" ")
            for t in token:
                if t not in vocab: 
                    vocab.append(t)
        # add 20 [UNUSED#1] tokens to the vocab
        for i in range(20):
            vocab.append(f"[UNUSED{i}]")
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        return self.vocab

    def load_vocab(self, vocab_path: str) -> None:
        print("Loading vocabulary from {}".format(vocab_path))
        with open(vocab_path, "r") as f:
            vocab = f.readlines()
        vocab = [token.strip("\n") for token in vocab]
        self.vocab = {token: idx for idx, token in enumerate(vocab)}

    def _tokenize(self, smi: str) -> str:
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|<|>|\*|\$|\%[0-9]{2}|[0-9])"  # type:ignore
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
        if self.vocab is None:
            raise ValueError("Vocabulary not found. Please build vocabulary first.")
        # encode tokens
        return [self.vocab[token] for token in token_list]

    def _batch_encode(self, tokens: list[str]) -> list[list[int]]:
        with Pool(processes=6) as pool:
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
    train_steps = math.ceil(batches_per_gpu / acc_batches) * (epochs+1)
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
    vocab_path = file_path / "data/vocab" / "chemspace_reduced" /"vocab_SMILES.txt"
    smiles_path = file_path / "data/databases" / "chemspace_reduced.txt"

    no_cores = os.cpu_count() - 2  # type: ignore
    examples = pd.read_csv(smiles_path, header=0)
    dataset_size = examples.shape[0]
    tokenizer = Tokenizer(no_cores, dataset_size)
    examples = examples["smi_can"].tolist()
    if not os.path.exists(vocab_path):
        tokenized = tokenizer.tokenize(examples)
        vocab = tokenizer.build_vocab(tokenized)
        os.makedirs(vocab_path.parent, exist_ok=True)
        with open(vocab_path, "w") as f:
            for token, idx in vocab.items():
                f.write(f"{token}\n")
    else:
        tokenized = tokenizer.tokenize(examples)
    max_len = max(len(token.split()) for token in tokenized)
    # get index for this max_len
    index = [i for i, token in enumerate(tokenized) if len(token.split()) == max_len]
    encoded = tokenizer.encode(tokenized)
    # get the length of first string in encoded
    print(encoded[index[0]])
