from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset, Subset
from scipy.sparse import csr_matrix, vstack
from src.model_utils import Tokenizer


class FGDataset(Dataset):
    def __init__(
        self,
        price: FloatTensor,
        features: LongTensor | csr_matrix,
        counts: Optional[LongTensor],
        smiles: Optional[list[str]],
        vocab_path: Optional[Path],
        augment: bool = False,
    ):
        # * Features/counts are stored as sparse matrix | list[LongTensor]

        self.price = price
        self.features = features
        self.counts = counts
        self.smiles = smiles
        self.augment = augment
        if augment:
            if vocab_path is None:
                raise ValueError("Vocab path is required for augmentation")
            if smiles is None:
                raise ValueError("Smiles are required for augmentation")
            if not isinstance(self.features, LongTensor):
                raise ValueError("Features must be LongTensor for augmentation")
            self.tokenizer = Tokenizer(1, 1)
            self.tokenizer.load_vocab(vocab_path=str(vocab_path))

    def augment_smiles(self, smi: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smi)
        random_smi = Chem.MolToSmiles(mol, doRandom=True)
        tokens = self.tokenizer.tokenize(random_smi)
        encoded = self.tokenizer.encode(tokens)
        return torch.LongTensor(encoded)

    def __len__(self):
        return len(self.price)

    def __getitem__(self, idx):
        if self.counts is None:
            return {"X": self.features[idx], "y": self.price[idx]}
        elif self.augment:
            X_aug = self.augment_smiles(self.smiles[idx])  # type: ignore
            return {
                "X": self.features[idx],
                "X_aug": X_aug,
                "y": self.price[idx],
                "y_aug": self.price[idx],
            }
        else:
            return {
                "X": self.features[idx],
                "y": self.price[idx],
                "c": self.counts[idx],
            }


class CombinedDataset(Dataset):
    def __init__(
        self,
        dataset_1: Subset[FGDataset],
        dataset_2: Subset[FGDataset],
    ):
        self.larger_dataset = (
            dataset_1 if len(dataset_1) > len(dataset_2) else dataset_2
        )
        self.smaller_dataset = (
            dataset_1 if len(dataset_1) < len(dataset_2) else dataset_2
        )
        self.combined_length = len(self.larger_dataset)
        self.small_length = len(self.smaller_dataset)
        
        # Access the underlying datasets
        larger_dataset_underlying = self.larger_dataset.dataset
        smaller_dataset_underlying = self.smaller_dataset.dataset
        
        if getattr(larger_dataset_underlying, 'augment', False) or getattr(smaller_dataset_underlying, 'augment', False):
            raise ValueError("Augmentation is not supported for combined datasets")
        if getattr(larger_dataset_underlying, 'counts', None) is not None:
            raise ValueError("Counts are not supported for combined datasets")

    def __len__(self):
        return self.combined_length

    def __getitem__(self, idx):
        large_data_point = self.larger_dataset[idx]
        small_idx = [i % self.small_length for i in idx]
        small_data_point = self.smaller_dataset[small_idx]
        
        x_lar, x_small = large_data_point["X"], small_data_point["X"]
        y_lar, y_small = large_data_point["y"], small_data_point["y"]
        
        if isinstance(x_lar, csr_matrix):
            X_con = vstack((x_lar, x_small))
        else:
            X_con = torch.cat((x_lar, x_small)) # type: ignore
        y_con = torch.cat((y_lar, y_small)) # type: ignore
        
        return {
            "X": X_con,
            "y": y_con
        }