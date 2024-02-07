import pandas as pd
import pickle 
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

with open("data/mport.pkl", "rb") as f:
    df_mport = pickle.load(f)

# apply MACCS keys to all smiles MACCSkeys.GenMACCSKeys(x)
def convert_to_maccs(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return fp 

def convert_to_maccs(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
    return fp

def process_smiles(smi):
    return convert_to_maccs(smi)

if __name__ == '__main__':
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_smiles, df_mport["smi_can"]), total=len(df_mport)))
        df_mport["maccs"] = results
        print(df_mport.head())

        # save pandas file 
        df_mport.to_pickle("data/mport_maccs.pkl")