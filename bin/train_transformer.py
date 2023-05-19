import numpy as np
import pickle
from tqdm import tqdm

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import r2_score

from src.model import TransformerEncoder,  EarlyStopping 
from src.data_loader import create_dataloader 
from src.definitions import CONFIGS_DIR, DATA_DIR

gin.external_configurable(MSELoss, module="torch.nn")
gin.external_configurable(optim.Adam, module="torch.optim")

@gin.configurable(denylist=["train_loader", "device", "model"])
def train(model: torch.nn.Module, optim: torch.optim.Optimizer, loss_func: torch.nn.modules.loss, train_loader, device):
    pass


# Create training function 

def train_trans(model: torch.nn.Module, optim: torch.optim.Optimizer, loss_func: torch.nn.modules.loss, scheduler, train_loader, device):
    model.to(device)

    optimizer = optim
    criterion = loss_func

    # Train the model
    train_loss = 0
    model.train()

    for data in tqdm(train_loader):
        optimizer.zero_grad()
        inputs, labels = data["X"], data["y"]
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        labels = labels
        output = output.view(-1,1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    return train_loss

@torch.no_grad()
def validate(model, val_loader, device):
    model.to(device)
    model.eval()
    output_list = []
    ground_truth = []
    
    for data in val_loader:
        inputs, labels = data["X"], data["y"]
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        output = output.view(-1,1)
        output_list.extend(output.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())
    output_list = np.array(output_list).reshape(-1,1)
    ground_truth = np.array(ground_truth).reshape(-1,1)
    # Calculate R2 score between predicted and true values
    r2 = r2_score(ground_truth, output_list)
    return r2


if __name__ == "__main__":
    # check cuda
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    gin.parse_config_file(CONFIGS_DIR.joinpath("train_configs.gin"))
    
    model_path = DATA_DIR.joinpath("model_data", "features_EFG_0.7.pkl")
    vocabs = pickle.load(open(DATA_DIR.joinpath("vocab","vocab_EFG_0.7.pkl"), "rb"))
    df_mport = pickle.load(open(DATA_DIR.joinpath("mport.pkl"), "rb"))
    features = pickle.load(open(model_path, "rb"))

    price, smiles = df_mport["price_mmol"].apply(np.log).tolist(), df_mport["smi_can"].tolist()
    train_loader, valid_loader, _ = create_dataloader(price, smiles, features)
    model = TransformerEncoder(input_size=len(vocabs)+2, embedding_size=128, num_layers=2, num_heads=4, hidden_size = 256)
    stopping = EarlyStopping()

    best_val = 0

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    warmup_steps = 1000
    total_steps = 10000

    # Create the learning rate scheduler
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.002, step_size_up=1000, cycle_momentum=False)
    loss_function = MSELoss()


    for epoch in tqdm(range(100)):
        train_loss = train_trans(model=model, train_loader=train_loader, device=device, optim=optimizer, loss_func=loss_function, scheduler=scheduler)
        val_r2 = validate(model=model, val_loader=valid_loader, device=device)
        print(f"Epoch: {epoch}, Train loss: {train_loss}")
        if val_r2 > best_val:
            best_val = val_r2
            torch.save(model.state_dict(), DATA_DIR.joinpath("model_data", "best_model.pt"))
            print(f"New best model saved with R2 score: {best_val}")
        
        if epoch % 5 == 0:
             print(f"Epoch: {epoch}, Train loss: {train_loss}, Val R2: {val_r2}")
        
        if stopping.early_stop(val_r2) and not stopping.deactivate:
            print("Early stopping")
            break