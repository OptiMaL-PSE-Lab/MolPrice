import numpy as np
import pickle

import gin
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from sklearn.metrics import r2_score

from src.model import FgLSTM
from src.data_loader import create_dataloader 
from src.definitions import DATA_DIR, CONFIGS_DIR


@torch.no_grad()
def test(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    ground_truth = []
    
    for data in test_loader:
        inputs, labels = data["X"], data["y"]
        inputs, labels = inputs.to(device), labels.to(device)
        output, perm_idx = model(inputs)
        labels = labels[perm_idx]
        output = output.view(-1,1)
        predictions.extend(output.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())
    predictions = np.array(predictions).reshape(-1,1)
    ground_truth = np.array(ground_truth).reshape(-1,1)
    return ground_truth, predictions


if __name__ == "__main__":
    # check cuda
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    gin.parse_config_file(CONFIGS_DIR.joinpath("test_configs.gin"))

    model_path = DATA_DIR.joinpath("model_data", "features_EFG.pkl")
    vocabs = pickle.load(open(DATA_DIR.joinpath("vocab","vocab_EFG.pkl"), "rb"))
    df_mport = pickle.load(open(DATA_DIR.joinpath("mport.pkl"), "rb"))
    features = pickle.load(open(model_path, "rb"))

    price, smiles = df_mport["price_mmol"].apply(np.log).tolist(), df_mport["smi_can"].tolist()
    _, _, test_loader = create_dataloader(price, smiles, features)
    
    # Load model from checkpoint
    model_path = DATA_DIR.joinpath("model_data", "best_model.pt")
    model = FgLSTM(input_size=len(vocabs)+2, count_size=2)
    model.load_state_dict(torch.load(model_path))
    ground_truth, predictions = test(model, test_loader, device)

    # Calculate R2 score between predicted and true values
    r2 = r2_score(ground_truth, predictions)
    
    # Scatter plot of predicted vs true values with heatmap of density and R2 score in legend
    plt.figure(figsize=(8,6))
    plt.hexbin(ground_truth, predictions, gridsize=100, cmap="viridis", bins="log")
    plt.colorbar()
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs True values")
    min_val, max_val = np.min([ground_truth, predictions]), np.max([ground_truth, predictions])
    plt.plot([min_val, max_val], [min_val, max_val],color='black')
    plt.legend(["R2 score: {:.3f}".format(r2)])
    plt.show()