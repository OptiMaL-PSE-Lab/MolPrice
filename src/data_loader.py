import pickle
import os
from abc import abstractmethod
from collections import Counter
from multiprocessing import Pool, Manager, cpu_count
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Generator, Callable

import datasets
import gin
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator  # type: ignore
from lightning.pytorch import LightningDataModule
from torch import LongTensor, FloatTensor
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack

from EFGs import mol2frag, cleavage
from src.rdkit_ifg import identify_functional_groups as ifg
from src.model_utils import Tokenizer, MolFeatureExtractor
from src.datasets_torch import FGDataset, CombinedDataset
from src.mhfp_encoder import MHFPEncoder


class CustomDataLoader(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        feature_path: Path,
        pickle_path: Path,
        batch_size: int,
        workers_loader: int,
        data_split: list[float],
        df_name: str,
        hp_tuning: bool,
        collate_func: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.feature_path = feature_path
        self.pickle_path = pickle_path
        self.batch_size = batch_size
        self.workers_loader = workers_loader
        self.data_split = data_split
        self.collate_fn = collate_func
        self.hp_tuning = hp_tuning
        self.mydataset: Dataset
        self.train_data: Subset
        self.val_data: Subset
        self.test_data: Subset

        self.dataframe: Path = self.data_path / df_name
        self.augment = False  # this is overwritten in TFLoader
        self.vocab_path: Path

    def prepare_data(self):
        if not self.pickle_path.exists() and not self.hp_tuning:
            self.feature_path.mkdir(parents=True, exist_ok=True)
            self.generate_features()
        elif self.hp_tuning:
            # TODO Only coded for fingerprints at the moment
            self.generate_features()
        else:
            pass

    def setup(self, stage: str) -> None:
        smiles = self.get_smiles() if self.augment else None
        self.price_dataset = FGDataset(
            *self.load_features(),
            smiles=smiles,
            vocab_path=self.vocab_path,
            augment=self.augment,
        )
        self.train_data, self.val_data, self.test_data = random_split(
            self.price_dataset,
            self.data_split,
            generator=torch.Generator().manual_seed(42),
        )
        if type(self).__name__ in ["TFLoader", "RoBERTaLoader"]:
            # sort data in train_data by length and shuffle indices for faster training due to different lengths
            initial_augment = self.augment
            self.train_data.dataset.augment = False  # type: ignore
            indices = sorted(
                range(len(self.train_data)),
                key=lambda x: len(self.train_data[x]["X"]),
                reverse=True,
            )
            # shuffle indices according to batch_size
            indices_groups = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]
            last_group = indices_groups.pop()
            new_indices = [
                item
                for sublist in np.random.permutation(indices_groups)
                for item in sublist
            ]
            new_indices.extend(last_group)
            self.train_data.dataset.augment = initial_augment  # type: ignore
            self.train_data = Subset(self.train_data, new_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers_loader,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.workers_loader,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.workers_loader,
            persistent_workers=False,
            collate_fn=self.collate_fn,
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
        if "price_mmol" in df.columns:
            return df["price_mmol"].apply(np.log).values  # type: ignore
        else:
            print(
                "WARNING: No price column found in dataframe. Using zeros instead. \n Note: Expected for combined datasets."
            )
            return np.zeros(df.shape[0])

    @abstractmethod
    def generate_features() -> None:
        pass

    @abstractmethod
    def load_features() -> (
        tuple[FloatTensor, LongTensor | csr_matrix, Optional[LongTensor]]
    ):
        pass


@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class EFGLoader(CustomDataLoader):
    def __init__(
        self,
        data_path,
        feature_path,
        batch_size,
        workers_loader,
        data_split,
        df_name,
        hp_tuning,
    ) -> None:
        pickle_path = feature_path / "features_EFG.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            workers_loader,
            data_split,
            df_name,
            hp_tuning,
        )
        self.vocab = Manager().dict()
        self.vocab_path = (
            data_path.parent / "vocab" / df_name.split(".")[0] / "vocab_EFG.pkl"
        )

    def load_features(self):
        # load from pickled features
        data = np.load(self.pickle_path, allow_pickle=True)
        fps = data["features"]
        price = data["price"]
        counts = data["counts"]
        fps = torch.from_numpy(fps).long()
        price = torch.from_numpy(price).float()
        counts = torch.from_numpy(counts).long()

        return price, fps, counts  # type: ignore

    def generate_features(self):
        # To create the feature vectors, there are 4 steps necesssary:
        # 1. Generate the vocabulary and reduce the size if desired
        # 2. Create the feature vectors based on the vocab
        # 3. Clean up empty features along with their pricing
        smiles = self.get_smiles()
        workers = cpu_count()
        # check if vocab already exists
        if self.vocab_path.exists():
            print("Loading vocabulary for EFGs")
            with open(self.vocab_path, "rb") as f:
                vocab = pickle.load(f)
            self.vocab = Manager().dict(vocab)
        else:
            print("Generating vocabulary for EFGs")
            with Pool(processes=workers) as pool:
                list(
                    tqdm(
                        pool.imap(
                            self.generate_vocab,
                            smiles,
                            chunksize=5 * workers,
                        ),
                        total=len(smiles),
                    )
                )
            vocab = dict(self.vocab)
            cleavage(vocab, alpha=0.7)
            self.vocab = Manager().dict(vocab)
            os.makedirs(self.vocab_path.parent, exist_ok=True)
            with open(self.vocab_path, "wb") as f:
                pickle.dump(vocab, f)

        print("Creating feature vectors for EFGs")
        with Pool(processes=workers) as p:
            features = list(
                tqdm(
                    p.imap(self._create_embeddings, smiles, chunksize=5 * workers),
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
        self,
        data_path,
        feature_path,
        batch_size,
        workers_loader,
        data_split,
        df_name,
        hp_tuning,
    ) -> None:
        pickle_path = feature_path / "features_IFG.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            workers_loader,
            data_split,
            df_name,
            hp_tuning,
        )
        self.vocab = Manager().dict()
        self.vocab_path = (
            data_path.parent / "vocab" / df_name.split(".")[0] / "vocab_IFG.pkl"
        )

    def load_features(self):
        # load from pickled features
        data = np.load(self.pickle_path, allow_pickle=True)
        features = data["features"]
        price = data["price"]
        counts = data["counts"]
        features = torch.from_numpy(features).long()
        price = torch.from_numpy(price).float()
        counts = torch.from_numpy(counts).long()

        return price, features, counts

    def generate_features(self):
        smiles = self.get_smiles()
        workers = cpu_count()

        if self.vocab_path.exists():
            print("Loading vocabulary for IFGs")
            with open(self.vocab_path, "rb") as f:
                vocab = pickle.load(f)
            self.vocab = Manager().dict(vocab)
        else:
            with Pool(workers) as p:
                # show progress bar with tqdm
                print("Generating vocabulary for IFGs")
                list(
                    tqdm(
                        p.imap(self._generate_vocab, smiles, chunksize=5 * workers),
                        total=len(smiles),
                    )
                )
                vocab = dict(self.vocab)
            with open(self.vocab_path, "wb") as f:
                pickle.dump(vocab, f)

        print("Creating feature vectors for IFGs")
        with Pool(workers) as p:
            features = list(
                tqdm(
                    p.imap(self._create_embeddings, smiles, chunksize=5 * workers),
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
        workers_loader,
        data_split,
        df_name,
        hp_tuning,
        fp_type: str,
        fp_size: int,
        p_r_size: int,
        two_d: bool,
        count_simulation: bool,
    ) -> None:
        if fp_type == "mhfp":
            pickle_path = feature_path / f"features_FP_{fp_type}_{fp_size}_False.npz"
        else: 
            pickle_path = (
                feature_path / f"features_FP_{fp_type}_{fp_size}_{count_simulation}.npz"
            )
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            workers_loader,
            data_split,
            df_name,
            hp_tuning,
            self.collate_fn,
        )

        self.fp_type = fp_type
        self.fp_size = fp_size  # the size of the fingerprint vector
        self.p_r_size = p_r_size  # the length of the path/radius
        self.two_d = two_d  # whether to use 2D info in addition to fingerprint
        self.count = count_simulation  # whether to use count fingerprint
        self.vocab_path = None  # type: ignore
        self.price_path = self.pickle_path.parent / "fp_prices.pkl.npz"

    def load_features(self):
        # load from pickled features
        price = np.load(self.price_path, allow_pickle=True)["price"]
        price = torch.from_numpy(price).float()
        fps = load_npz(self.pickle_path)
        if self.two_d:
            features = load_npz(self.pickle_path.parent / "features_2D.npz")
            fps = hstack([fps, features])
        return price, fps, None  # type: ignore

    def generate_features(self):
        if os.path.exists(self.pickle_path):
            return

        print(f"Creating feature vectors for {self.fp_type} fingerprint")
        # * The higher the fp_size, the larger the memory requirements -> use sparse vector object instead
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
        elif self.fp_type == "mhfp":
            pass
        else:
            raise ValueError("Fingerprint type not supported")

        if self.fp_type == "mhfp":
            with Pool(os.cpu_count()) as p:
                fps = list(
                    tqdm(
                        p.imap(self._mhfp_features, self.get_smiles(), chunksize=20),
                        total=len(self.get_smiles()),
                    )
                )
        else:
            fps = []
            for smi in self.get_batch_smiles(100000):
                for s in smi:
                    mol = Chem.MolFromSmiles(s)  # type: ignore
                    fp = self.fp_gen.GetFingerprintAsNumPy(mol)
                    fps.append(fp)

        fps = np.array(fps, dtype=np.uint8)
        fps = csr_matrix(fps)
        two_d_path = self.pickle_path.parent / "features_2D.npz"
        if self.two_d and not two_d_path.exists():
            feature_extractor = MolFeatureExtractor(self.feature_path)
            features = feature_extractor.encode(self.get_smiles())
            features = np.array(features)
            features = feature_extractor.standardise_features(features)
            features = csr_matrix(features)
            # concatenate the two sparse matrices
            save_npz(two_d_path, features)

        save_npz(self.pickle_path, fps)
        if not self.price_path.exists():
            price = self.get_price()
            np.savez_compressed(self.price_path, price=price, allow_pickle=True)

    def _mhfp_features(self, smi: str) -> np.ndarray:
        return MHFPEncoder.secfp_from_smiles(smi, length=self.fp_size)

    # * Overwrite due to sparse matrix manipulation needed to load in FPs
    def train_dataloader(self) -> DataLoader:
        sampler = BatchSampler(
            RandomSampler(self.train_data), batch_size=self.batch_size, drop_last=False
        )
        return DataLoader(
            self.train_data,
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = BatchSampler(
            RandomSampler(self.val_data), batch_size=self.batch_size, drop_last=False
        )
        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = BatchSampler(
            RandomSampler(self.test_data), batch_size=self.batch_size, drop_last=False
        )
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # batch is sparse csr matrix -> convert to dense tensor
        data_batch = batch[0]["X"]
        if type(data_batch) == csr_matrix:
            data_batch = data_batch.tocoo()
            values = data_batch.data
            indices = np.array((data_batch.row, data_batch.col))
            shape = data_batch.shape
            i, v, s = (
                torch.LongTensor(indices),
                torch.FloatTensor(values),
                torch.Size(shape),  # type: ignore
            )
            X = torch.sparse.FloatTensor(i, v, s)  # type: ignore
            X = X.to_dense()
        else:
            raise ValueError("Data type not supported")

        return {"X": X, "y": batch[0]["y"]}  # type: ignore


# Data Loader for creating Tokens from SMILES strings
@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class TFLoader(CustomDataLoader):
    def __init__(
        self,
        data_path,
        feature_path,
        batch_size,
        workers_loader,
        data_split,
        df_name,
        hp_tuning,
        augment,
    ) -> None:
        pickle_path = feature_path / "features_TF.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            workers_loader,
            data_split,
            df_name,
            hp_tuning,
            self.collate_fn,
        )
        self.vocab_path = (
            data_path.parent / "vocab" / df_name.split(".")[0] / "vocab_SMILES.txt"
        )
        self.augment = augment

    def generate_features(self):
        print("Creating features for Tokenized SMILES")
        smiles = self.get_smiles()
        workers = cpu_count()
        tokenizer = Tokenizer(workers, len(smiles))
        print("Tokenizing dataset...")
        tokenized = tokenizer.tokenize(smiles)  # returns list[int] of length nxSMILES
        if not self.vocab_path.exists():
            vocab = tokenizer.build_vocab(tokenized)
            os.makedirs(self.vocab_path.parent, exist_ok=True)
            with open(self.vocab_path, "w") as f:
                for token in vocab.keys():
                    f.write(f"{token}\n")
        else:
            tokenizer.load_vocab(str(self.vocab_path))
        del smiles
        print("Encoding tokens...")
        encoded = tokenizer.encode(
            tokenized
        )  # returns list[list[int]] of length n x variable_length
        encoded = np.array(encoded, dtype="object")
        price = self.get_price()
        np.savez_compressed(self.pickle_path, price=price, features=encoded)

    def load_features(self):
        data = np.load(self.pickle_path, allow_pickle=True)
        features = data["features"]
        price = data["price"]
        price = torch.from_numpy(price).float()
        features = [torch.Tensor(row) for row in features]  # * tensors not padded

        return price, features, None

    def collate_fn(self, batch):
        """Takes list of tensors and pads them to same length for batching"""
        data_batch = [b["X"] for b in batch]
        y = torch.stack([b["y"] for b in batch])
        if self.augment:
            data_aug = [b["X_aug"] for b in batch]
            data_batch.extend(data_aug)
            y_aug = torch.stack([b["y_aug"] for b in batch])
            y = torch.cat([y, y_aug])
        data_batch = pad_sequence(data_batch, batch_first=True, padding_value=0)
        data_batch = data_batch.long()
        return {"X": data_batch, "y": y}

@gin.configurable(denylist=["data_path", "feature_path"])  # type: ignore
class RoBERTaLoader(CustomDataLoader):
    def __init__(
        self,
        data_path,
        feature_path,
        batch_size,
        workers_loader,
        data_split,
        df_name,
        hp_tuning,
    ) -> None:
        pickle_path = feature_path / "features_BERT.pkl.npz"
        super().__init__(
            data_path,
            feature_path,
            pickle_path,
            batch_size,
            workers_loader,
            data_split,
            df_name,
            hp_tuning,
            self.collate_fn
        )
        self.vocab_path = None  # type: ignore
        self.tokenizer: RobertaTokenizer
        self.price: np.ndarray
        self.features: np.ndarray


    def generate_features(self):
        #* as datagenerating is quick, no need to save data
        smiles = self.get_smiles()
        smiles = {"smi": smiles}
        custom_dataset = datasets.Dataset.from_dict(smiles)
        self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
        encoded = custom_dataset.map(self._tokenize, batched=True, num_proc=self.workers_loader, batch_size=10000)
        self.encoded = encoded["input_ids"]
        price = self.get_price()
        self.price = price

    def load_features(self):
        price = torch.from_numpy(self.price).float()
        features = [torch.Tensor(row) for row in self.encoded]
        return price, features, None
        

    def _tokenize(self, examples):
        return self.tokenizer(examples["smi"], padding="do_not_pad", truncation=True)

    def collate_fn(self, batch):
        data_batch = [b["X"] for b in batch]
        y = torch.stack([b["y"] for b in batch])
        data_batch = pad_sequence(data_batch, batch_first=True, padding_value=0)
        data_batch = data_batch.long()
        return {"X": data_batch, "y": y}

# Data Loader explicitly used for testing on diverse datasets
class TestLoader(LightningDataModule):
    def __init__(
        self,
        test_file: Path,
        data_path: Path,
        batch_size: int,
        model_name: str,
        has_price: bool,
    ):
        super().__init__()
        self.test_file = test_file
        self.data_path = data_path
        self.batch_size = batch_size
        self.model_name = model_name
        self.has_price = has_price
        self.fp_size: int

    def prepare_data(self):

        # query standard configs from gin
        df_name = gin.query_parameter("%df_name")
        data_split = gin.query_parameter("%data_split")

        # get smiles from data frame
        smiles = self.get_smiles()

        if self.model_name == "Fingerprint":
            fp_type = gin.query_parameter("FPLoader.fp_type")
            p_r_size = gin.query_parameter("FPLoader.p_r_size")
            count_sim = gin.query_parameter("FPLoader.count_simulation")
            self.fp_size = gin.query_parameter("%fp_size")
            two_d = gin.query_parameter("FPLoader.two_d")
            print(f"Creating feature vectors for {fp_type} fingerprint")

            if fp_type == "morgan":
                # * The higher the fp_size, the larger the memory requirements -> use sparse vector object instead
                self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=p_r_size, fpSize=self.fp_size, countSimulation=count_sim
                )
            elif fp_type == "rdkit":
                self.fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
                    maxPath=p_r_size, fpSize=self.fp_size, countSimulation=count_sim
                )
            elif fp_type == "atom":
                self.fp_gen = rdFingerprintGenerator.GetAtomPairGenerator(
                    maxDistance=p_r_size,
                    fpSize=self.fp_size,
                    countSimulation=count_sim,
                )

            fps = []

            if fp_type == "mhfp":
                with Pool(10) as p:
                    fps = list(
                        tqdm(
                            p.imap(self._mhfp_features, smiles, chunksize=20),
                            total=len(smiles),
                        )
                    )
            else:
                for s in tqdm(smiles):
                    mol = Chem.MolFromSmiles(s)  # type: ignore
                    fp = self.fp_gen.GetFingerprintAsNumPy(mol)
                    fps.append(fp)

            if two_d:
                feature_extractor = MolFeatureExtractor(self.test_file.parent.parent)
                features = feature_extractor.encode(self.get_smiles())
                features = np.array(features)
                features = feature_extractor.standardise_features(features)
                # concatenate the two sparse matrices
                fps = np.hstack([fps, features])

            fps = torch.FloatTensor(fps)
            fps = [fps, None]

        elif self.model_name == "Transformer":
            print("Creating features for Tokenized SMILES")
            workers = cpu_count()
            tokenizer = Tokenizer(workers, len(smiles))
            print("Tokenizing dataset...")
            tokenized = tokenizer.tokenize(
                smiles
            )  # returns list[int] of length nxSMILES
            vocab_path = (
                self.data_path.parent
                / "vocab"
                / df_name.split(".")[0]
                / "vocab_SMILES.txt"
            )
            tokenizer.load_vocab(vocab_path)
            print("Encoding tokens...")
            fps = tokenizer.encode(tokenized)
            fps = torch.LongTensor(fps)
            fps = [fps, None]

        elif self.model_name[:4] == "LSTM":
            current_dataset = df_name.split(".")[0]
            if self.model_name == "LSTM_EFG":
                lstm_loader = EFGLoader(
                    self.data_path,
                    self.data_path,
                    64,
                    10,
                    data_split,
                    current_dataset,
                    False,
                )
            elif self.model_name == "LSTM_IFG":
                lstm_loader = IFGLoader(
                    self.data_path,
                    self.data_path,
                    64,
                    10,
                    data_split,
                    current_dataset,
                    False,
                )
            else:
                raise ValueError("Model not supported")

            workers = lstm_loader.workers_loader
            with Pool(workers) as p:
                features = list(
                    tqdm(
                        p.imap(
                            lstm_loader._create_embeddings,
                            smiles,
                            chunksize=5 * workers,
                        ),
                        total=len(smiles),
                    )
                )

            #! watch out for this possible error
            # price, features = zip(*[(price[idx], features[idx]) for idx, i in enumerate(features) if i])
            # count the occurence of features
            features = [Counter(f) for f in features]
            counts = [torch.Tensor(list(f.values())) for f in features]
            features = [torch.Tensor(list(f.keys())) for f in features]
            padded_features = pad_sequence(features, batch_first=True, padding_value=0)
            padded_counts = pad_sequence(counts, batch_first=True, padding_value=0)
            padded_features = padded_features.long()
            padded_counts = padded_counts.long()

            fps = [padded_features, padded_counts]

        return fps

    def test_dataloader(self) -> DataLoader:
        print("preparing features for testing")
        features = self.prepare_data()
        first_feat = features[0]
        if self.has_price:
            df = pd.read_csv(self.test_file)
            price = df["price"].apply(np.log).to_list()
        else:
            price = torch.FloatTensor(torch.zeros(first_feat.shape[0]))
        test_data = FGDataset(price, *features, vocab_path=None, smiles=None)  # type: ignore

        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            num_workers=10,
            persistent_workers=True,
        )

    def get_smiles(self) -> list[str]:
        df = pd.read_csv(self.test_file)
        return df["smi_can"].to_list()

    def _mhfp_features(self, smi: str) -> np.ndarray:
        return MHFPEncoder.secfp_from_smiles(smi, length=self.fp_size)


class CombinedLoader(LightningDataModule):
    """
    This loader takes in two loaders and combines their data into one dataloader using the CombinedDataset
    """

    def __init__(self, loader_1: CustomDataLoader, loader_2: CustomDataLoader) -> None:
        super().__init__()
        self.loader_1 = loader_1
        self.loader_2 = loader_2
        self.loaders = [self.loader_1, self.loader_2]
        # check if loaders are of same class
        if type(self.loader_1) != type(self.loader_2):
            raise ValueError("Loaders must be of same class")

    def prepare_data(self):
        for loader in self.loaders:
            if not loader.pickle_path.exists():
                loader.feature_path.mkdir(parents=True, exist_ok=True)
                loader.generate_features()
            else:
                pass

    def setup(self, stage: str):
        train_list, val_list, test_list = [], [], []
        for loader in self.loaders:
            smiles = loader.get_smiles() if loader.augment else None
            dataset = FGDataset(
                *loader.load_features(),
                smiles=smiles,
                vocab_path=loader.vocab_path,
                augment=loader.augment,
            )
            train_data, val_data, test_data = random_split(
                dataset, loader.data_split, generator=torch.Generator().manual_seed(42)
            )
            if type(loader).__name__ == "TFLoader":
                # sort data in train_data by length and shuffle indices for faster training due to different lengths
                indices = sorted(
                    range(len(train_data)),
                    key=lambda x: len(train_data[x]["X"]),
                    reverse=True,
                )
                # shuffle indices according to batch_size
                indices_groups = [
                    indices[i : i + loader.batch_size]
                    for i in range(0, len(indices), loader.batch_size)
                ]
                last_group = indices_groups.pop()
                new_indices = [
                    item
                    for sublist in np.random.permutation(indices_groups)
                    for item in sublist
                ]
                new_indices.extend(last_group)
                train_data = Subset(train_data, new_indices)

            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)

        self.loader_1.train_data = CombinedDataset(*train_list)  # type: ignore
        self.loader_1.val_data = CombinedDataset(*val_list)  # type: ignore
        self.loader_1.test_data = CombinedDataset(*test_list)  # type: ignore

    def train_dataloader(self) -> DataLoader:
        return self.loader_1.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.loader_1.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.loader_1.test_dataloader()


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / "data"
    feature_path = data_path / "features"
    df_name = "chemspace_reduced.txt"
    batch_size = 32
    workers_loader = 8
    data_split = [0.8, 0.1, 0.1]
    efgloader = IFGLoader(
        data_path,
        feature_path,
        batch_size,
        workers_loader,
        data_split,
        df_name,
        hp_tuning=False,
    )

    efgloader.generate_features()
