# pyright: reportAttributeAccessIssue=false
import pickle
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


# convert prices from price_mmol to price_g
def converter_price(smi, price_mmol):
    mol = Chem.MolFromSmiles(smi)
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    price_g = price_mmol * 1000 / mw  # going from $/mmol to $/g
    return price_g


def get_prices(df_molport: pd.DataFrame):
    # drop index column from df_molport
    df_molport.drop(df_molport.columns[0], axis=1)
    smiles = df_molport["smi_can"].to_list()
    prices_mmol = df_molport["price_mmol"].to_list()
    with Pool(8) as p:
        prices_g = list(
            tqdm(
                p.starmap(converter_price, tqdm(zip(smiles, prices_mmol), total=len(smiles), chunksize=16)
            ))
        )
    new_df = {"SMILES": smiles, "price": prices_g}
    new_df = pd.DataFrame(new_df)

    return new_df

if __name__ == "__main__":
    data_file = pd.read_csv("data/databases/molport_reduced.txt")
    pickle_file = pickle.load(open("indices.pkl", "rb"))
    df_coprinet = get_prices(data_file)
    train_indices = pickle_file["train_indices"]
    val_indices = pickle_file["val_indices"]
    test_indices = pickle_file["test_indices"]
    df_train = df_coprinet.iloc[train_indices]
    df_val = df_coprinet.iloc[val_indices]
    df_test = df_coprinet.iloc[test_indices]
    df_train.to_csv("train_coprinet.csv", index=False)
    df_val.to_csv("val_coprinet.csv", index=False)
    df_test.to_csv("test_coprinet.csv", index=False)
    
