# MolPrice

A deep learning model for synthetic accessibility based on molecular prices.

## Installation

```bash
# Get the code
git clone https://github.com/yourusername/MolPrice.git
cd MolPrice

# Create environment
conda env create -f molprice.yml
conda activate molprice

```

## Making Predictions

```bash
# Single molecule prediction
python predict.py --mol "CC(=O)OC1=CC=CC=C1C(=O)O" --model latest

# Batch prediction
python predict.py --mol molecules.csv --smiles-col SMILES --output predictions.csv
```

## Reproducing Results

```bash
# Download datasets
python scripts/get_data.py

# Train model
python train.py --config configs/default.yaml

# Generate evaluation plots
python evaluate.py --model-path outputs/best_model.pt
```
