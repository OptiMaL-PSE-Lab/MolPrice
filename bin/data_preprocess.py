# pyright: reportAttributeAccessIssue=false

import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors 

from src.definitions import ROOT_DIR, DATA_DIR

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

class Preprocessing:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.num_workers: int =  os.cpu_count() - 2 # type: ignore
        self.chunk_size = self.num_workers * 2
        self.data_frame: pd.DataFrame

    def _canonicalize_and_convert(self, smiles: str, price: float, unit: str, amount: float
    ) -> tuple[Optional[str], Optional[float]]:
        smi = smiles.split(".")[-1]
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        if not mol:
            return (None, None)

        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        m_weight = rdMolDescriptors.CalcExactMolWt(mol)
        # m_weight has units of g/mol
        if unit == "mg":
            m_weight = m_weight * 1e3 / amount
        elif unit == "ml":
            # assume water density of solvent
            m_weight = m_weight * 1e3 / amount
        elif unit == "g":
            m_weight = m_weight / amount
        elif unit == "ug":
            m_weight = m_weight * 1e6 / amount
        else:
            raise ValueError("Unit not recognized")
        
        price *= m_weight / 1e3  # price has units of $/mmol
            
        return (smi, price)

    def extract_chemspace(self):
        chemspace_path = self.data_path / "chemspace_data.smiles"
        df_chemspace = pd.read_csv(chemspace_path, sep="\t", header=0)
        # extract relevant columns of df into four lists of names
        task = zip(df_chemspace["SMILES"], df_chemspace["Price_EUR"], df_chemspace["Units"], df_chemspace["Pack"])
        pool = Pool(processes=self.num_workers)

        results = pool.starmap(
            self._canonicalize_and_convert,
            tqdm(task, total=df_chemspace.shape[0]),
            chunksize=self.chunk_size
        )
        smiles, new_price = zip(*results)
        df_chemspace["smi_can"] = smiles
        df_chemspace["price_mmol"] = new_price
        df_chemspace = df_chemspace.dropna()
        df_chemspace.to_csv(str(self.data_path / "chemspace_data.csv"))
        return df_chemspace
    
    def reduce_size(self, df: pd.DataFrame, lower_bound: float) -> None:
        """Reduce size of dataframe to a specified price"""
        df = df[df["price_mmol"].apply(np.log) > lower_bound]
        df = df.drop(df.columns[0], axis=1) 
        df.to_csv(str(self.data_path / "chemspace_reduced.csv"), index=False)
        self.data_frame = df


    def plot_price_distribution(self, chemspace_df: pd.DataFrame) -> None:
        """Plot distribution of prices in chemspace dataset"""
        if self.data_frame is not None:
            chemspace_df = self.data_frame
        prices = chemspace_df["price_mmol"].apply(np.log).values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
        ax.hist(prices, bins=100, range=(5,20)) # type:ignore
        ax.set_xlabel("Price ($/mmol)")
        ax.set_ylabel("Frequency")
        # Create title for overall figure
        fig.suptitle("Distribution of Pricing Data", fontsize=14)
        fig.savefig(str(ROOT_DIR.joinpath("price_dist.png")))
        print(chemspace_df.nlargest(10, "price_mmol")[["smi_can", "price_mmol"]].to_dict())
        

# run extract chemspace
if __name__ == "__main__":
    data_path = DATA_DIR
    preprocessor = Preprocessing(data_path)
    # chemspace_df = preprocessor.extract_chemspace()
    chemspace_df = pd.read_csv(data_path / "chemspace_data.csv")
    preprocessor.reduce_size(chemspace_df, 8.5)
    preprocessor.plot_price_distribution(chemspace_df)
