# pyright: reportAttributeAccessIssue=false

import itertools
import math as mt
import os
import tempfile
import shutil
import warnings
from abc import abstractmethod
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Deque, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path(__file__).parent.parent


class Preprocessing:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.num_workers: int = os.cpu_count() - 2  # type: ignore
        self.chunk_size = self.num_workers * 2
        self.data_frame: pd.DataFrame

    def _canonicalize_and_convert(
        self, smiles: str, price: float, unit: str, amount: float
    ) -> tuple[Optional[str], Optional[float]]:
        try:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        except:
            mol = None
        if not mol:
            return (None, None)

        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        m_weight = rdMolDescriptors.CalcExactMolWt(mol)
        m_weight = self._weight_converter(m_weight, amount, unit)

        if m_weight:
            price *= m_weight / 1e3  # convert to $/mmol
        else:
            return (None, None)

        return (smi, price)  # * Price is in $/mmol

    def _weight_converter(
        self, m_weight: float, amount: float, unit: str
    ) -> Optional[float]:
        # m_weight has units of g/mol
        # check if amount is 0
        if amount == 0:
            return None

        unit_converter = {
            "mg": 1e3,
            "ml": 1,
            "g": 1,
            "ug": 1e6,
            "micromol": 1e6,
        }  # * Most important units in database, discard kg and L do to errors in data
        if unit in unit_converter:
            m_weight = m_weight * unit_converter[unit] / amount
        else:
            return None

        return m_weight
    
    @staticmethod
    def reduce_size(
        lower_bound: float, df: pd.DataFrame, name: str, data_path: Path
    ) -> None:
        """Reduce size of dataframe to a specified price"""
        df_new = df[df["price_mmol"] > lower_bound]
        df_new = df_new.drop(df.columns[0], axis=1)
        df_new.to_csv(data_path / f"{name}_reduced.txt", index=True
        )
        print(f"Reduced size of {name} dataset from {df.shape[0]} to {df_new.shape[0]} molecules")

    @staticmethod
    def plot_price_distribution(price_df: pd.DataFrame, dataset_name: str) -> None:
        """Plot distribution of prices in single dataset"""
        prices = price_df["price_mmol"].apply(np.log).values.tolist()
        weights = np.ones_like(prices) * 100 / len(prices)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), layout="constrained")
        ax.hist(
            prices,
            bins=50,
            weights=weights,
            range=(2, max(prices) - 3),
            color="tab:red",
            alpha=0.45,
            edgecolor="k",
            lw=1,
        )  # type:ignore
        ax.set_xlabel(r"log Price $(\$/mmol)$", fontsize=13)
        ax.set_ylabel(r"Frequency (%)", fontsize=13)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Create title for overall figure
        fig.savefig(
            str(ROOT_DIR.joinpath("figs", f"price_dist_{dataset_name}.png")), dpi=600
        )
        fig.savefig(
            str(ROOT_DIR.joinpath("figs", "svgs", f"price_dist_{dataset_name}.svg")),
            dpi=600,
        )
        with open(
            ROOT_DIR.joinpath("figs", f"price_dist_{dataset_name}.txt"), "w"
        ) as f:
            f.write(f"\nTop 50 most expensive molecules in {dataset_name}\n")
            f.write("-" * 50 + "\n")
            f.write(
                str(price_df.nlargest(10000, "price_mmol")[["smi_can", "price_mmol"]])
                + "\n"
            )
            f.write(f"\nTop 50 cheapest molecules in {dataset_name}\n")
            f.write("-" * 50 + "\n")
            f.write(
                str(price_df.nsmallest(50, "price_mmol")[["smi_can", "price_mmol"]])
                + "\n"
            )
            f.write("-" * 50 + "\n")

    @staticmethod
    def plot_overlap_distribution(price_df: dict[str, pd.DataFrame]) -> None:
        """Plot overlapping distribution of prices in multiple datasets"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), layout="constrained")
        default_colour = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
        ]
        for i, (key, df) in enumerate(price_df.items()):
            prices = df["price_mmol"].apply(np.log).values.tolist()
            weights = np.ones_like(prices) * 100 / len(prices)
            ax.hist(
                prices,
                bins=50,
                weights=weights,
                range=(1, 12),
                label=key,
                color=default_colour[i],
                alpha=0.4,
                lw=1,
                edgecolor="black",
            )  # type:ignore

        ax.set_xlabel(r"log Price $(\$/mmol)$", fontsize=13)
        ax.set_ylabel(r"Frequency (%)", fontsize=13)
        ax.legend(fontsize=12, loc="upper left")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Create title for overall figure
        fig.savefig(str(ROOT_DIR.joinpath("figs", "distribution_overlap.png")))
        fig.savefig(str(ROOT_DIR.joinpath("figs", "svgs", "distribution_overlap.svg")))

    @abstractmethod
    def extract_data(self):
        """
        To be called by child classes to extract pricing info from the respective datasets
        """
        pass


class ChemspaceExtractor(Preprocessing):
    def __init__(self, data_path, csv_name: str, chunk_size: None):
        super().__init__(data_path)
        self.csv_name = csv_name

    def extract_data(self):
        chemspace_path = self.data_path / self.csv_name
        df_chemspace = pd.read_csv(chemspace_path, sep="\t", header=0)
        # extract relevant columns of df into four lists of names
        task = zip(
            df_chemspace["SMILES"],
            df_chemspace["Price_EUR"],
            df_chemspace["Units"],
            df_chemspace["Pack"],
        )
        pool = Pool(processes=self.num_workers)

        results = pool.starmap(
            self._canonicalize_and_convert,
            tqdm(task, total=df_chemspace.shape[0]),
            chunksize=self.chunk_size,
        )
        smiles, new_price = zip(*results)
        df_chemspace["smi_can"] = smiles
        df_chemspace["price_mmol"] = new_price
        df_chemspace = df_chemspace.dropna()
        df_chemspace.to_csv(str(self.data_path / "chemspace_prices.txt"))
        return df_chemspace


class MolportExtractor(Preprocessing):
    def __init__(self, data_path, csv_name: str, chunk_size: int = 5000):
        super().__init__(data_path)
        self.csv_name = csv_name
        self.chunk_size = chunk_size

    def extract_data(self):
        # * Large datafile with 5M molecules and 110M rows
        # Expect headers and dtype for df_molport
        headers = {
            "MOLECULE_ID": "int32",
            "CD_SMILES": "str",
            "PRICEPERUNIT": "float16",
            "SELLUNIT": "float16",
            "MEASURE": "category",
        }
        temp_dir = tempfile.mkdtemp(dir=self.data_path)
        molport_path = self.data_path / self.csv_name
        df_molport = pd.read_csv(
            molport_path, usecols=list(headers.keys()), dtype=headers
        )
        df_molport.dropna(inplace=True)
        indices = self._get_indices(df_molport)

        print(
            "Extracting best pricing for each molecule in chunks of {} chunks per iteration".format(
                self.chunk_size
            )
        )
        i = 0
        for batch_indices in tqdm(
            self._batch_indices_generator(indices),
            total=mt.ceil(len(indices) / self.chunk_size),
        ):

            tasks = [
                (
                    df_molport["CD_SMILES"][start:end].tolist(),
                    df_molport["PRICEPERUNIT"][start:end].tolist(),
                    df_molport["MEASURE"][start:end].tolist(),
                    df_molport["SELLUNIT"][start:end].tolist(),
                    df_molport["MOLECULE_ID"][start:end].tolist(),
                )
                for start, end in batch_indices
            ]

            with Pool(processes=self.num_workers) as pool:
                results = list(
                    pool.starmap(
                        self._obtain_best_pricing,
                        tasks,  # task should be a list of tuples of
                        chunksize=5 * self.num_workers,
                    )
                )

            ids, smiles, prices, units = zip(*results)
            df_new_molport = pd.DataFrame(
                {
                    "MOLECULE_ID": ids,
                    "SMILES": smiles,
                    "PRICEPERUNIT": prices,
                    "ORIGINAL_UNIT": units,
                }
            )
            df_new_molport.dropna(inplace=True)
            ids, smiles, prices, units = (
                df_new_molport["MOLECULE_ID"],
                df_new_molport["SMILES"],
                df_new_molport["PRICEPERUNIT"],
                df_new_molport["ORIGINAL_UNIT"],
            )

            # write each line of the dataframe to temp text file
            with open(os.path.join(temp_dir, f"temp_{i}.txt"), "w") as f:
                for i, s, p, u in zip(ids, smiles, prices, units):
                    f.write(f"{i},{s},{p}, {u}\n")

            i += 1

        with open(self.data_path / "molport_prices.txt", "w") as outfile:
            outfile.write("id,smi_can,price_mmol,orig_units\n")
            for filename in os.listdir(temp_dir):
                with open(os.path.join(temp_dir, filename), "r") as infile:
                    outfile.write(infile.read())

        shutil.rmtree(temp_dir)

        self.data_frame = pd.read_csv(self.data_path / "molport_prices.txt")

    def _get_indices(self, df_molport: pd.DataFrame) -> Deque[tuple[int, int]]:
        # Get indices of molecules with multiple prices
        count_df = df_molport.groupby("MOLECULE_ID")  # type:ignore
        count_df = count_df.count()
        # get indices for same molecules
        counting_index = -1
        index_list = deque()
        print("Obtaining indices for molecules with multiple prices")
        for i in tqdm(range(len(count_df))):
            current_index = count_df.iloc[i, 0] + counting_index  # type:ignore
            index_list.append((counting_index + 1, current_index)) # type:ignore
            counting_index = current_index

        return index_list

    def _batch_indices_generator(
        self, indices: Deque[tuple[int, int]]
    ) -> Generator[list[tuple[int, int]], None, None]:
        it = iter(indices)
        while True:
            batch_indices = list(itertools.islice(it, self.chunk_size))
            if not batch_indices:
                return
            yield batch_indices

    def _obtain_best_pricing(
        self,
        smiles: list[str],
        prices: list[float],
        units: list[str],
        amounts: list[float],
        ids: list[int],
    ) -> tuple[Optional[int], Optional[str], Optional[float], Optional[str]]:
        # Obtain best pricing for each molecule
        incumbant_price = 1e20
        new_unit = None
        new_price = None

        for smi, pri, un, am in zip(smiles, prices, units, amounts):
            smi = smi.split("|")[0]
            # split smiles based on
            new_smi, new_price = self._canonicalize_and_convert(smi, pri, un, am)
            if not new_price:
                continue
            elif new_price < incumbant_price:
                incumbant_price = new_price
                new_unit = un

        if not new_price:
            return (None, None, None, None)
        else:
            return (
                ids[0],
                new_smi,
                float(incumbant_price),
                new_unit,
            )  # * Price is in $/mmol


# run extract chemspace
if __name__ == "__main__":
    from argparse import ArgumentParser

    DEFAULT_TASK = "extract"
    DEFAULT_DATA_DIR = "data/databases"
    DEFAULT_CHUNK_SIZE = 25000
    DEFAULT_DATASETS = ["chemspace_data.smiles", "molport_data.smiles"]
    DEFAULT_PRICE_THRESHOLD = 2

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data file from root directory",
        default=DEFAULT_DATA_DIR,
    )
    arg_parser.add_argument(
        "--task",
        type=str,
        choices=["extract", "plot", "reduce"],
        help="Whether to plot the data distribution, extract data from database or reduce size of existing dataset",
        default=DEFAULT_TASK,
    )
    arg_parser.add_argument(
        "--dataset",
        help="Name of dataset to extract data from e.g. chemspace.smiles",
        nargs="+",
        default=DEFAULT_DATASETS,
    )
    arg_parser.add_argument(
        "--price_threshold","--pt", type=float, default=DEFAULT_PRICE_THRESHOLD
    )
    arg_parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size for processing data",
        default=DEFAULT_CHUNK_SIZE,
    )
    args = arg_parser.parse_args()

    avaiable_extractors = {"chem": ChemspaceExtractor, "molp": MolportExtractor}
    data_path = ROOT_DIR.joinpath(args.data_dir)

    if args.task == "extract":
        print("Extracting data from database(s)...")
        def check_extractor(dataset: str) -> None:
            if dataset[:4] not in avaiable_extractors:
                raise ValueError(f"Dataset Extractor {args.dataset} not available")

        if len(args.dataset) > 1:
            for db in args.dataset:
                check_extractor(db)
                extractor = avaiable_extractors[db[:4]](data_path, db, args.chunk_size)
                extractor.extract_data()
        else:
            db = args.dataset[0]
            check_extractor(db)
            extractor = avaiable_extractors[db[:4]](
                data_path, db, args.chunk_size
            )
            extractor.extract_data()

    elif args.task == "plot":
        print("Plotting price distribution(s)...")
        if len(args.dataset) > 1:
            for i, db in enumerate(args.dataset):
                db_name = db.split("_")[0]
                df_i = pd.read_csv(data_path / db)
                Preprocessing.plot_price_distribution(df_i, db_name)

            Preprocessing.plot_overlap_distribution(
                {db.split("_")[0]: pd.read_csv(data_path / db) for db in args.dataset}
            )
        else:
            db = args.dataset[0]
            df = pd.read_csv(data_path / db)
            Preprocessing.plot_price_distribution(df, db.split("_")[0])
    
    elif args.task == "reduce":
        print("Reducing size of dataset(s)...")
        if len(args.dataset) > 1:
            for db in args.dataset:
                df = pd.read_csv(data_path / db)
                Preprocessing.reduce_size(args.price_threshold, df, db.split("_")[0], data_path)
        else:
            db = args.dataset[0]
            df = pd.read_csv(data_path / db)
            Preprocessing.reduce_size(args.price_threshold, df, db.split("_")[0], data_path)