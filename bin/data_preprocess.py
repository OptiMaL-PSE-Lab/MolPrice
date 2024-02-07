import os
import pickle
import warnings
from abc import abstractmethod
from multiprocessing import Pool, Manager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

from src.rdkit_ifg import identify_functional_groups as ifg
from src.definitions import ROOT_DIR, DATA_DIR
from EFGs import mol2frag, cleavage

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

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
    fig.savefig(ROOT_DIR.joinpath("figs", "price_dist.png"))
    
class Preprocessing:
    def __init__(self, col_names: list[str] ,data_path: Path, reduce_size: bool):
        self.data_path = data_path
        self.reduce_size = reduce_size
        self.num_workers = os.cpu_count() - 2
        self.col_names = col_names

    def _canonicalize_and_convert(self, smiles, price, unit, amount):
        smi = smiles.split('.')[-1]
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return (None, None)
        
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        m_weight = rdMolDescriptors.CalcExactMolWt(mol)
        # m_weight has units of g/mol
        if unit == 'mg':
            m_weight = m_weight * 1e-3 * amount
        elif unit == 'g':
            m_weight = m_weight * amount
        elif unit == 'ug':
            m_weight = m_weight * 1e-6 * amount
        else:
            raise ValueError(f"Invalid unit {unit}")
        
        price /= m_weight

        return (smi, price)
    
    

        




