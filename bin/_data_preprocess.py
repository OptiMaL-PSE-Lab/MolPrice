import os
import pickle
import warnings
from abc import abstractmethod
from multiprocessing import Pool, Manager
from pathlib import Path

import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors # type:ignore
from tqdm import tqdm

from src.rdkit_ifg import identify_functional_groups as ifg
from src.definitions import ROOT_DIR, DATA_DIR
from EFGs import mol2frag, cleavage
# Import from data_dist to plot distribution of prices

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

columns = [
    "smi",
    "smi_can",
    "id",
    "inchi",
    "inchi_key",
    "iupac",
    "pubchem",
    "ver_amount",
    "unver_amount",
    "is_sc",
    "is_bb",
    "comp_state",
    "qc_meth",
    "lead_time",
    "price_1mg",
    "price_5mg",
    "price_50mg",
    "price_100mg",
    "price_250mg",
    "price_1g",
]

def plot_price_distribution(path_mport: Path):
    """Plot distribution of prices in mport dataset"""
    mport = pd.read_pickle(path_mport)
    old_prices, prices = mport["price_100mg"], mport["price_mmol"]
    fig,[ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.hist(np.log(prices), bins=100)
    ax1.set_xlabel('Log Price ($/mmol)')
    ax1.set_ylabel('Frequency')
    #ax1.set_xlim((1.5,5))
    ax2.hist(np.log(old_prices), bins=100)
    ax2.set_xlabel('Log Price ($/100mg)')
    ax2.set_ylabel('Frequency')
    #ax2.set_xlim((2.5,5.5))
    # Create title for overall figure
    fig.suptitle('Distribution of Pricing Data')
    fig.savefig(str(ROOT_DIR.joinpath("figs", "price_dist.png")))
    


class Preprocessing:
    def __init__(
        self, colnames: list, data_path: Path, data_smiles: Path, reduce_size: bool
    ):
        self.colnames = colnames
        self.data_smiles = data_smiles
        self.reduce_size = reduce_size
        self.num_workers = os.cpu_count() - 1 # type:ignore
        self.vocab = Manager().dict()
        self.data_path = data_path
        self.vocab_path = self.data_path.joinpath("vocab")
        
        if not self.data_smiles.exists():
            self.extract_pubchem() if "pubchem"==self.data_smiles.stem else self.extract_mport()

        check_path = self.vocab_path / f"vocab_{type(self).__name__}.pkl"
        if check_path.exists():
            with check_path.open("rb") as f:
                vocab = pickle.load(f)
                self.vocab.update(vocab)
        else:
            self.generate_vocab()

    def _read_file(self, file_path):
        with gzip.open(file_path, "rb") as f:
            data = pd.read_csv(
                f, sep="\t", names=self.colnames, header=None, skiprows=1
            )
        return data

    def _range_to_value(self, range_str):
        split_str = range_str.split()
        numbers = [float(s) for s in split_str if s.isdigit()]
        try:
            return sum(numbers) / len(numbers)
        except ZeroDivisionError:
            return None

    def _canonicalize_and_convert(self, smi, price):
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return (None, None)
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            # Do parsing twice as rdkit sometimes fails to produce valid smiles
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            m_weight = rdMolDescriptors.CalcExactMolWt(mol)
            # Convert prices from $/100mg to $/mmol
            price *= np.array(m_weight) / 100
            return (smi, price)
        except Exception:
            return (None, None)

    def extract_mport(self):
        mport_path = self.data_path / "mport_data"
        files = list(mport_path.iterdir())
        pool = Pool(processes=self.num_workers)
        all_moles = pool.map(self._read_file, files)
        df_data = pd.concat(all_moles)
        df_data = df_data.iloc[:, [1, -3]]
        df_data = df_data.dropna(subset=["price_100mg"])
        df_data["price_100mg"] = df_data[["price_100mg"]].applymap(self._range_to_value)
        print("Only retain smiles that can be read into rdkit:")
        task = zip(df_data["smi_can"], df_data["price_100mg"])
        results = pool.starmap(
            self._canonicalize_and_convert,
            tqdm(task, total=df_data.shape[0]),
            chunksize=int(2 * self.num_workers),
        )
        smi, price = zip(*results)
        df_data["smi_can"] = smi
        df_data["price_mmol"] = price
        df_data.dropna(axis=0, how="any", inplace=True)

        with self.data_smiles.open("wb") as f:
            pickle.dump(df_data, f)

    def extract_pubchem(self):
        pubchem_file = self.data_path / "CID-SMILES.gz"
        smiles = []
        print("Extracting smiles from pubchem...")
        with gzip.open(pubchem_file, "rb") as f:
            for line in f:
                line = line.decode("utf-8")
                contents = line.split()
                smiles.append(contents[1])
            print("Finished extracting smiles from pubchem")
        df_pubchem = pd.DataFrame(smiles, columns=["smi_can"])
        with self.data_smiles.open("wb") as f:
            pickle.dump(df_pubchem, f)

    def generate_vocab(self):
        smiles_dataset = self.get_smiles()
        print(f"Generating vocabulary for {type(self).__name__}...")
        with Pool(processes=self.num_workers) as pool:
            list(
                tqdm(
                    pool.imap(
                        self.convert_mol_to_frag,
                        smiles_dataset,
                        chunksize=5 * self.num_workers,
                    ),
                    total=len(smiles_dataset),
                )
            )
        if self.reduce_size:
            if type(self).__name__ == "EFG":
                # Calculate percentage of vocab that occur at least 2 times in dataset
                vocab = dict(self.vocab)
                #res = [(tool, value) for tool, value in vocab.items() if value >= 2]
                #alpha = len(res) / len(self.vocab)
                cleavage(vocab, alpha=0.7)
            else:
                # This is not encouraged as it will remove a lot of information
                vocab = {k: v for k, v in self.vocab.items() if v >= 2}
            
            self.vocab = Manager().dict()
            self.vocab.update(vocab)

        pkl_path = self.vocab_path / f"vocab_{type(self).__name__}.pkl"
        # Convert to normal dict
        vocab = dict(self.vocab)
        with pkl_path.open("wb") as f:
            pickle.dump(vocab, f)

    def create_feature_vec(self):
        smiles_dataset = self.get_smiles(True)
        print(f"Creating feature vectors for {type(self).__name__}...")
        with Pool(processes=self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        self._create_embeddings,
                        smiles_dataset,
                        chunksize=2 * self.num_workers,
                    ),
                    total=len(smiles_dataset),
                )
            )

        # results will be 2d list where dim 1 is no smiles and dim 2 is variable length list of indeces
        # Save list as pickle
        pkl_path = self.data_path / 'model_data' / f"features_{type(self).__name__}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(results, f)

    def get_smiles(self, call=False):
        if self.data_smiles.suffix == ".pkl" or call:
            with self.data_smiles.open("rb") as f:
                df_data = pickle.load(f)
                smi = df_data["smi_can"].tolist()
                return smi
        else:
            df_data = pd.read_csv(self.data_smiles)
            return df_data["smi_can"].tolist()

    @abstractmethod
    def _create_embeddings(self):
        pass

    @abstractmethod
    def convert_mol_to_frag(self):
        pass

    @abstractmethod
    def reduce_vocab_size(self):
        pass


class IFG(Preprocessing):
    def __init__(
        self, colnames: list, data_path: Path, data_smiles: Path, reduce_size=False
    ):
        super().__init__(colnames, data_path, data_smiles, reduce_size)

    def convert_mol_to_frag(self, smile):
        mol = Chem.MolFromSmiles(smile)
        fgs = ifg(mol)
        for fg in fgs:
            self.vocab[fg.atoms] = self.vocab.get(fg.atoms, 0) + 1

    def _create_embeddings(self, smile):
        """ 
        Returns a list of indeces of positions in vocab
        """
        vocab = list(self.vocab().keys())
        mol = Chem.MolFromSmiles(smile)
        ifg_list = ifg(mol)
        atoms = [fg.atoms for fg in ifg_list]
        indeces = []
        for atom in atoms:
            if atom in vocab:
                idx = vocab.index(atom)
                indeces.append(idx+1)
            else:
                idx = len(vocab)
                indeces.append(idx+1)

        return indeces


class EFG(Preprocessing):
    def __init__(
        self, colnames: list, data_path: Path, data_smiles: Path, reduce_size=False
    ):
        super().__init__(colnames, data_path, data_smiles, reduce_size)

    def convert_mol_to_frag(self, smile):
        try:
            mol = Chem.MolFromSmiles(smile)
            a, b = mol2frag(mol)
            vocab_update = {}
            for elem in a + b:
                vocab_update[elem] = self.vocab.get(elem, 0) + 1
            self.vocab.update(vocab_update)
        except:
            pass
            # These are errors with the efg algorithm rather than rdkit
    
    def _create_embeddings(self, smile):
       try: 
            vocab = list(self.vocab.keys())
            indeces = []
            mol = Chem.MolFromSmiles(smile)
            extra = {}
            a, b = mol2frag(mol, toEnd=True, vocabulary=list(vocab), extra_included=True, extra_backup=extra)
            backup_vocab = list(extra.values())
            atoms = a + b + backup_vocab
            for atom in atoms:
                if atom in vocab:
                    idx = vocab.index(atom)
                    indeces.append(idx+1)
                else:
                    idx = len(vocab)
                    indeces.append(idx+1)
            return indeces
       except:
           # These are errors with the efg algorithm rather than rdkit
           return []


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent / "data"

    # efg = EFG(columns, DATA_DIR, DATA_DIR / "chemspace_reduced.csv", True)
    # efg.create_feature_vec()

    ifg = IFG(columns, DATA_DIR, DATA_DIR / "chemspace_reduced.csv", False)
    ifg.create_feature_vec()
    
    # Plot price distribution
    #plot_price_distribution(DATA_DIR / "mport.pkl")
