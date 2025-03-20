<div style="float:right; margin-left:20px; margin-top: -30px;">
    <img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptiMLPSE" height="150" align="right"/>
</div>
<br>
<br>

# MolPrice
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning model for synthetic accessibility prediction based on molecular prices.

## Installation
Clone the repository and create a virtual environment with conda:
```bash
# Get the code
git clone https://github.com/fredhastedt/MolPrice.git
cd MolPrice

# Create environment
conda env create -f molprice.yml
conda activate molprice

```
We provide model checkpoints for MolPrice via Figshare **xyz**. One can choose from the following models: 
<br>
1. SECFP fingerprint (with or w/o 2D features)
2. Morgan Fingerprint (with or w/o 2D features)

Once the model is downloaded, place in **./models** directory.

## Making Price Predictions
One can run the code per molecule or using batch prediction. In case of batch prediction, please first save all molecules in a .csv file.

```bash
# Single molecule prediction
python -m bin.predict --mol "CC(=O)OC1=CC=CC=C1C(=O)O" --cn MP_SECFP_hybrid

# Batch prediction
python -m bin.predict --mol molecules.csv --cn MP_SECFP_hybrid --smiles-col SMILES_COLUMN
```

## Reproducing SA Test Results
The test datasets for SA comparison can be obtained from Figshare via **xyz**. Once the files are downloaded, place within **./testing** directory.
<br>
The results for each test dataset can be obtained by running: 
```bash
python -m bin.test main_ood --model Fingerprint --cn MODEL_CHECKPOINT --test_name TEST_FILE1,TEST_FILE2 --combined
```
For example, if one downloaded the MP_SECFP_hybrid model and saved the test files 3 as follows: TS3_hs.csv and TS3_es.csv, once can run: 
```bash
python -m bin.test main_ood --model Fingerprint --cn MP_SECFP_hybrid/best.ckpt --test_name TS3_hs.csv,TS3_es.csv --combined
```

## Model Training
If one has access to a database containing molecules along with their prices, one can run the following script to train their own model (given that prices are in log(USD)/mmol): 

```bash
python -m bin.train --model MODEL_TYPE --fp FINGERPRINT_TYPE
```

Within the script, the following arguments can be adjusted: 
    - **model**: Choose between [Fingerprint, RoBERTa, Transformer, LSTM_EFG]
    - **fp**: Choose between [atom, rdkit, morgan, mhfp] (mhfp is the SECFP fingerprint encoder)
    
In one has a pre-trained Fingerprint model, one can train the model on the contrastive loss by calling: 
```bash
python -m bin.train --model Fingerprint --fp FINGERPRINT_TYPE --combined --cn MODEL_CHECKPOINT
```

