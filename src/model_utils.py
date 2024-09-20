import gin
import math
import re
import os
import pandas as pd
from gin import query_parameter as gin_qp
from tqdm import tqdm
from typing import Optional
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, SpacialScore


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


class MolFeatureExtractor:
    def __init__(self):
        pass

    def encode(self, smi: str | list[str]) -> list[tuple] | tuple:
        if isinstance(smi, list):
            return self._batch_encode(smi)
        elif isinstance(smi, str):
            return self._encode(smi)

    def _encode(self, smi: str) -> tuple:
        return self._calculate_2D_feat(smi)

    def _batch_encode(self, smiles: list[str]) -> list[tuple]:
        with Pool(processes=10) as pool:
            encoded = list(
                tqdm(
                    pool.imap(self._encode, smiles, chunksize=20),
                    total=len(smiles),
                )
            )
        return encoded

    def _calculate_2D_feat(self, smi):
        mol = Chem.MolFromSmiles(smi)
        sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        sps = SpacialScore.SPS(mol)
        stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        heterocyc = rdMolDescriptors.CalcNumHeterocycles(mol)
        no_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        no_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_macro, n_multi = self.numMacroAndMulticycle(mol, mol.GetNumAtoms())
        return (
            sp3,
            sps,
            stereo,
            rot_bonds,
            tpsa,
            heterocyc,
            no_spiro,
            no_bridgehead,
            n_macro,
            n_multi,
        )

    def numMacroAndMulticycle(self, mol, nAtoms):
        ri = mol.GetRingInfo()  # type: ignore
        nMacrocycles = 0
        multi_ring_atoms = {i: 0 for i in range(nAtoms)}
        for ring_atoms in ri.AtomRings():
            if len(ring_atoms) > 6:
                nMacrocycles += 1
            for atom in ring_atoms:
                multi_ring_atoms[atom] += 1
        nMultiRingAtoms = sum([v - 1 for k, v in multi_ring_atoms.items() if v > 1])
        return nMacrocycles, nMultiRingAtoms


def calculate_max_training_step(path_data) -> None:
    batch_size, acc_batches, epochs = (
        gin_qp("TFLoader.batch_size"),
        gin_qp("main.gradient_accum"),
        gin_qp("main.max_epoch"),
    )
    if not isinstance(batch_size, int):
        batch_size = gin_qp("%batch_size")
    dataframe_name, splits = gin_qp("%df_name"), gin_qp("%data_split")
    df = pd.read_csv(path_data / dataframe_name)
    len_train = df.shape[0] * splits[0]
    batches_per_gpu = math.ceil(len_train / float(batch_size))
    train_steps = math.ceil(batches_per_gpu / acc_batches) * epochs
    gin.bind_parameter(
        "transformer/torch.optim.lr_scheduler.OneCycleLR.total_steps", train_steps
    )


if __name__ == "__main__":
    import os
    import pandas as pd
    from pathlib import Path

    file_path = Path(__file__).parent.parent
    vocab_path = file_path / "data/vocab" / "chemspace_reduced" / "vocab_SMILES.txt"
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
