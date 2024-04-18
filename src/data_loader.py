import pickle
from abc import abstractmethod
from collections import Counter
from multiprocessing import Pool, Manager
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Generator

import gin
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator  # type: ignore
from lightning.pytorch import LightningDataModule
from torch import LongTensor, FloatTensor
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence

from EFGs import mol2frag, cleavage
from src.rdkit_ifg import identify_functional_groups as ifg
from src.model_utils import Tokenizer


class FGDataset(Dataset):
    def __init__(
        self, price: FloatTensor, features: LongTensor, counts: Optional[LongTensor]
    ):
        self.price = price
        self.features = features
        self.counts = counts

    def __len__(self):
        return len(self.price)

    def __getitem__(self, idx):
        if self.counts is None:
            return {"X": self.features[idx], "y": self.price[idx]}
        else:
            return {
                "X": self.features[idx],
                "y": self.price[idx],
                "c": self.counts[idx],
            }


class CustomDataLoader(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        feature_path: Path,
        pickle_path: Path,
        batch_size: int,
        num_workers: int,
        data_split: list[float],
        df_name: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.feature_path = feature_path
        self.pickle_path = pickle_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_split = data_split
        self.mydataset: Dataset
        self.train_data: Subset
        self.val_data: Subset
        self.test_data: Subset

        self.dataframe: Path = self.data_path / f"{df_name}.csv"

    def prepare_data(self):
        if not self.pickle_path.exists():
            self.feature_path.mkdir(parents=True, exist_ok=True)
            self.generate_features()
        else:
            pass

    def setup(self, stage: str) -> None:
        # dimensions are: price (n, 1), features (n, m), counts (n, m)
        self.mydataset = FGDataset(*self.load_features())
        self.train_data, self.val_data, self.test_data = random_split(
            self.mydataset, self.data_split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    # * holding df in memory is slowing down pool -> use getter instead
    def get_smiles(self) -> list[str]:
        df = pd.read_csv(self.dataframe)
        return df["smi_can"].tolist()

    def get_batch_smiles(self, batch_size: int) -> Generator[list[str], None, None]:
        df = pd.read_csv(self.dataframe)
        for i in tqdm(range(0, len(df), batch_size)):
            yield df["smi_can"].iloc[i : i + batch_size].tolist()

    def get_price(self) -> np.ndarray:
        """
        Returns the log price of the molecules / mmol
        """
        df = pd.read_csv(self.dataframe)
        return df["price_mmol"].apply(np.log).values  # type: ignore

    @abstractmethod
    def generate_features() -> None:
        pass

    @abstractmethod
    def load_features() -> tuple[FloatTensor, LongTensor, Optional[LongTensor]]:
        pass


@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class EFGLoader(CustomDataLoader):
    def __init__(
        self, data_path, feature_path, batch_size, num_workers, data_split, df_name
    ) -> None:
        pickle_path = feature_path / "features_EFG.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            num_workers,
            data_split,
            df_name,
        )
        self.vocab = Manager().dict()
        self.vocab_path = data_path / "vocab" / "vocab_EFG.pkl"

    def load_features(self):
        # load from pickled features
        data = np.load(self.pickle_path, allow_pickle=False)
        fps = data["features"]
        price = data["price"]
        counts = data["counts"]
        fps = torch.from_numpy(fps).long()
        price = torch.from_numpy(price).float()
        counts = torch.from_numpy(counts).long()

        return price, fps, counts # type: ignore

    def generate_features(self):
        # To create the feature vectors, there are 4 steps necesssary:
        # 1. Generate the vocabulary and reduce the size if desired
        # 2. Create the feature vectors based on the vocab
        # 3. Clean up empty features along with their pricing
        smiles = self.get_smiles()
        # check if vocab already exists
        if self.vocab_path.exists():
            print("Loading vocabulary for EFGs")
            with open(self.vocab_path, "rb") as f:
                vocab = pickle.load(f)
            self.vocab = Manager().dict(vocab)
        else:
            print("Generating vocabulary for EFGs")
            with Pool(processes=self.num_workers) as pool:
                list(
                    tqdm(
                        pool.imap(
                            self.generate_vocab,
                            smiles,
                            chunksize=5 * self.num_workers,
                        ),
                        total=len(smiles),
                    )
                )
            vocab = dict(self.vocab)
            cleavage(vocab, alpha=0.7)
            self.vocab = Manager().dict(vocab)
            with open(self.vocab_path, "wb") as f:
                pickle.dump(vocab, f)

        print("Creating feature vectors for EFGs")
        with Pool(processes=self.num_workers) as p:
            features = list(
                tqdm(
                    p.imap(
                        self._create_embeddings, smiles, chunksize=5 * self.num_workers
                    ),
                    total=len(smiles),
                )
            )

        # get price from dataframe
        price = self.get_price()
        # now remove empty features
        price, features = zip(
            *[(price[idx], features[idx]) for idx, i in enumerate(features) if i]
        )
        # count the occurence of features
        features = [Counter(f) for f in features]
        counts = [torch.Tensor(list(f.values())) for f in features]
        features = [torch.Tensor(list(f.keys())) for f in features]
        padded_features = pad_sequence(features, batch_first=True, padding_value=0)
        padded_counts = pad_sequence(counts, batch_first=True, padding_value=0)
        # padded_counts and padded_features have same dimension
        # convert to LongTensor
        padded_features = padded_features.long()
        padded_counts = padded_counts.long()
        # convert tensors to numpy for easier saving
        padded_features, padded_counts = padded_features.numpy(), padded_counts.numpy()

        # store info in dictionary for easy access
        np.savez_compressed(
            self.pickle_path,
            price=price,
            features=padded_features,
            counts=padded_counts,
        )

    def generate_vocab(self, smile: str) -> None:
        try:
            mol = Chem.MolFromSmiles(smile)  # type: ignore
            a, b = mol2frag(mol)
            vocab_update = {}
            for elem in a + b:
                vocab_update[elem] = self.vocab.get(elem, 0) + 1
            self.vocab.update(vocab_update)
        except:
            pass

    def _create_embeddings(self, smi: str) -> list[Optional[int]]:
        try:
            vocab = list(self.vocab.keys())
            mol = Chem.MolFromSmiles(smi)  # type: ignore
            extra = {}
            a, b = mol2frag(
                mol,
                toEnd=True,
                vocabulary=list(vocab),
                extra_included=True,
                extra_backup=extra,
            )
            backup_vocab = list(extra.values())
            atoms = a + b + backup_vocab
            indeces = []
            for atom in atoms:
                if atom in vocab:
                    idx = vocab.index(atom)
                    indeces.append(idx + 1)  # add 1 for padding
                else:
                    idx = len(vocab)
                    indeces.append(idx + 1)
            return indeces
        except:
            return []


@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class IFGLoader(CustomDataLoader):
    def __init__(
        self, data_path, feature_path, batch_size, num_workers, data_split, df_name
    ) -> None:
        pickle_path = feature_path / "features_IFG.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            num_workers,
            data_split,
            df_name,
        )
        self.vocab = Manager().dict()
        self.vocab_path = data_path / "vocab" / "vocab_IFG.pkl"

    def load_features(self):
        # load from pickled features
        data = np.load(self.pickle_path, allow_pickle=False)
        features = data["features"]
        price = data["price"]
        counts = data["counts"]
        features = torch.from_numpy(features).long()
        price = torch.from_numpy(price).float()
        counts = torch.from_numpy(counts).long()

        return price, features, counts

    def generate_features(self):
        smiles = self.get_smiles()

        if self.vocab_path.exists():
            print("Loading vocabulary for IFGs")
            with open(self.vocab_path, "rb") as f:
                vocab = pickle.load(f)
            self.vocab = Manager().dict(vocab)
        else:
            with Pool(self.num_workers) as p:
                # show progress bar with tqdm
                print("Generating vocabulary for IFGs")
                list(
                    tqdm(
                        p.imap(
                            self._generate_vocab, smiles, chunksize=5 * self.num_workers
                        ),
                        total=len(smiles),
                    )
                )
                vocab = dict(self.vocab)
            with open(self.vocab_path, "wb") as f:
                pickle.dump(vocab, f)

        print("Creating feature vectors for IFGs")
        with Pool(self.num_workers) as p:
            features = list(
                tqdm(
                    p.imap(
                        self._create_embeddings, smiles, chunksize=5 * self.num_workers
                    ),
                    total=len(smiles),
                )
            )

        # get price from dataframe
        price = self.get_price()
        # now remove empty features
        price, features = zip(
            *[(price[idx], features[idx]) for idx, i in enumerate(features) if i]
        )
        # count the occurence of features
        features = [Counter(f) for f in features]
        counts = [torch.Tensor(list(f.values())) for f in features]
        features = [torch.Tensor(list(f.keys())) for f in features]
        padded_features = pad_sequence(features, batch_first=True, padding_value=0)
        padded_counts = pad_sequence(counts, batch_first=True, padding_value=0)
        padded_features = padded_features.long()
        padded_counts = padded_counts.long()
        padded_features, padded_counts = padded_features.numpy(), padded_counts.numpy()

        # store info in dictionary for easy access
        np.savez_compressed(
            self.pickle_path,
            price=price,
            features=padded_features,
            counts=padded_counts,
        )

    def _generate_vocab(self, smi: str) -> None:
        mol = Chem.MolFromSmiles(smi)  # type: ignore
        fgs = ifg(mol)
        for fg in fgs:
            self.vocab[fg.atoms] = self.vocab.get(fg.atoms, 0) + 1

    def _create_embeddings(self, smi: str) -> list[int]:
        """
        Returns a list of indeces of positions in vocab
        """
        vocab = list(self.vocab.keys())  # type: ignore
        mol = Chem.MolFromSmiles(smi)  # type: ignore
        ifg_list = ifg(mol)
        atoms = [fg.atoms for fg in ifg_list]
        indeces = []
        for atom in atoms:
            if atom in vocab:
                idx = vocab.index(atom)
                indeces.append(idx + 1)
            else:
                idx = len(vocab)
                indeces.append(idx + 1)

        return indeces


# The loader for fingerprints of the molecules - most fingerprints share same signature in rdkit
@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class FPLoader(CustomDataLoader):
    def __init__(
        self,
        data_path,
        feature_path,
        batch_size,
        num_workers,
        data_split,
        df_name,
        fp_type: str,
        fp_size: int,
        p_r_size: int,
        count_simulation: bool,
    ) -> None:
        pickle_path = feature_path / f"features_FP_{fp_type}.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            num_workers,
            data_split,
            df_name,
        )
        self.fp_type = fp_type
        self.fp_size = fp_size  # the size of the fingerprint vector
        self.p_r_size = p_r_size  # the length of the path/radius
        self.count = count_simulation  # whether to use count fingerprint

    def load_features(self) -> tuple[FloatTensor, FloatTensor, Optional[LongTensor]]:
        # load from pickled features
        data = np.load(self.pickle_path, allow_pickle=False)
        fps = data["features"]
        price = data["price"]
        fps = torch.from_numpy(fps).float()
        price = torch.from_numpy(price).float()

        return price, fps, None  # type: ignore

    def generate_features(self):
        print(f"Creating feature vectors for {self.fp_type} fingerprint")
        # * The higher the fp_size, the larger the memory requirements -> use batch processing to only load in part of SMILES at each it
        if self.fp_type == "morgan":
            self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.p_r_size, fpSize=self.fp_size, countSimulation=self.count
            )
        elif self.fp_type == "rdkit":
            self.fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                maxPath=self.p_r_size, fpSize=self.fp_size, countSimulation=self.count
            )
        elif self.fp_type == "atom":
            self.fp_gen = rdFingerprintGenerator.GetAtomPairGenerator(
                maxDistance=self.p_r_size,
                fpSize=self.fp_size,
                countSimulation=self.count,
            )
        else:
            raise ValueError("Fingerprint type not supported")
        fps = []
        for smi in self.get_batch_smiles(10000):
            for s in smi:
                mol = Chem.MolFromSmiles(s)  # type: ignore
                fp = self.fp_gen.GetFingerprintAsNumPy(mol)
                fps.append(fp)

        fps = np.array(fps, dtype=np.int8)
        price = self.get_price()
        np.savez_compressed(
            self.pickle_path, price=price, features=fps, allow_pickle=False
        )


# Data Loader for creating Tokens from SMILES strings
@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class TFLoader(CustomDataLoader):
    def __init__(
        self,
        data_path,
        feature_path,
        batch_size,
        num_workers,
        data_split,
        df_name,
    ) -> None:
        pickle_path = feature_path / "features_TF.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            num_workers,
            data_split,
            df_name,
        )

    def generate_features(self):
        print("Creating features for Tokenized SMILES")
        vocab_path = self.data_path / "vocab" / "vocab_SMILES.txt"
        with open(vocab_path, "r") as f:
            vocab = {line.strip(): idx for idx, line in enumerate(f)}
        smiles = self.get_smiles()
        tokenizer = Tokenizer(vocab, 700, self.num_workers, len(smiles))
        tokenized = tokenizer.tokenize(smiles)  # returns list[int] of length nxSMILES
        encoded = tokenizer.encode(
            tokenized
        )  # returns list[list[int]] of length n x max_len x 1
        encoded = np.array(encoded)
        price = self.get_price()
        np.savez_compressed(self.pickle_path, price=price, features=encoded)

    def load_features(self):
        data = np.load(self.pickle_path, allow_pickle=False)
        features = data["features"]
        price = data["price"]
        features = torch.from_numpy(features).long()
        price = torch.from_numpy(price).float()

        return price, features, None


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / "data"
    feature_path = data_path / "features"
    df_name = "chemspace_reduced"
    batch_size = 32
    num_workers = 8
    data_split = [0.8, 0.1, 0.1]
    efgloader = IFGLoader(
        data_path,
        feature_path,
        batch_size,
        num_workers,
        data_split,
        df_name
    )

    efgloader.generate_features()
