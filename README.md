# MolPrice

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
python -m bin.predict --mol molecules.csv --cn MP_SECFP_hybrid --smiles-col xyz
```

## Reproducing Results
### Reproducing Test-Set Results
The test datasets for SA comparison can be obtained from Figshare via **xyz**.

### Model Training
If one has access to a database containing molecules along with their prices, one can run the following scripts to train their own model:

