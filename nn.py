from torch import nn
import torch
import pickle 
import pandas
from tqdm import tqdm
with open("data\mport_maccs.pkl","rb") as f:
    df_mport = pickle.load(f)

class MACCFingerprint(nn.Module):
    def __init__(self):
        super(MACCFingerprint, self).__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(167, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.neural_network(x)
        return x
    
    def predict(self, x):
        return self.forward(x)

import numpy as np
# Get X features and Y outputs from df_mport
X = np.array(df_mport["maccs"].values.tolist())
Y = df_mport["price_mmol"].tolist()
# Tensorize X and Y
X = torch.from_numpy(X).float()
Y = torch.tensor(Y)

# Split data into train and test randomly
from sklearn.model_selection import train_test_split
# import r2_score
from sklearn.metrics import r2_score
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)

# Instantiate model
model = MACCFingerprint()

# Instantiate loss function
loss_fn = nn.MSELoss()
# Instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# start training with 100 epochs and batch size of 32

r2_best = 0

# find cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")

else:
    device = torch.device("cpu")

def train(model, X_train, Y_train, X_test, Y_test, loss_fn, optimizer, epochs=100, batch_size=32, device=device):
    # move model to device
    model.to(device)
    for epoch in tqdm(range(epochs)):
        # shuffle data
        perm_idx = torch.randperm(X_train.size()[0])
        # get batch size
        batch_size = batch_size
        # get number of batches
        num_batches = X_train.size()[0] // batch_size
        # initialize loss
        loss = 0
        for i in range(num_batches):
            # get batch
            batch_idx = perm_idx[i*batch_size:(i+1)*batch_size]
            # get X and Y batch
            X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
            # zero out gradients
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            # get output from model
            output = model(X_batch)
            # calculate loss
            loss = loss_fn(output, Y_batch.view(-1,1))
            # backpropagate loss
            loss.backward()
            # update weights
            optimizer.step()
        # print loss every 10 epochs
        if epoch % 10 == 0:
            print("Epoch: {0}, Loss: {1}".format(epoch, loss.item()))
            # calculate r2 score on test set
            model.eval()
            Y_test_pred = model.predict(X_test.to(device))
            r2_test = r2_score(Y_test, Y_test_pred.detach().cpu().numpy())
            print("R2 score Test: {0}".format(r2_test))
            # test on test set and save model if r2 score is better than previous best
            if r2_test > r2_best:
                torch.save(model.state_dict(), "data/model_data/maccs_model.pt")
    return model


model = train(model, X_train, Y_train, X_test, Y_test, loss_fn, optimizer, epochs=100, batch_size=32)
