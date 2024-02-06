# Load data into correct pytorch dataloader 
from collections import Counter

import gin
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split 
from src.definitions import CONFIGS_DIR

# One function to remove empty rows from the data and deletes according smiles

def pad_features(features, counts):
    """ 
    Create a function to convert a list of variable-length features to a padded sequence
    """
    # Convert features to a list of tensors and get their lengths
    features = [torch.LongTensor(feature) for feature in features]
    counts = [torch.LongTensor(count) for count in counts]

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_counts = pad_sequence(counts, batch_first=True, padding_value=0)

    return padded_features, padded_counts


def remove_empty_features(price, smiles, features):
    """ 
    Takes a list of features and records indeces where all features are zero. 
    Removes these indeces from both the features tensor and price tensor
    """
    if len(smiles) != len(features):
        raise ValueError("Smiles and features must be the same length")
    
    price, smiles, features = zip(*[(price[idx], smiles[idx], features[idx]) for idx, i in enumerate(features) if i])
    return price, smiles,features

def feat_and_count(features):
    """ 
    Takes a list of features and counts the number of each feature
    """
    features = [Counter(f) for f in features]
    count = [list(f.values()) for f in features]
    features = [list(f.keys()) for f in features]
    return features, count

class FGDataset(Dataset):
    def __init__(self, features,counts, price, smiles):
        price = torch.FloatTensor(price)
        price = price.view(-1,1)
        self.smiles = smiles
        self.c = counts
        self.X = features
        self.y = price
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {"X":self.X[idx], "c": self.c[idx], "y": self.y[idx]}

@gin.configurable(denylist=["price", "smiles", "features"])
def create_dataloader(price, smiles, features, batch_size: int, train_split: float, test_split:float, shuffle: bool, num_workers:int):
    # Create a train and test set
    price, smiles, features = remove_empty_features(price, smiles, features)
    features, counts = feat_and_count(features)
    features, counts = pad_features(features, counts)
    X_train, X_test, counts_train, counts_test, smi_train, smi_test, y_train, y_test = train_test_split(features, counts, smiles, price, train_size=train_split, test_size=test_split,random_state=42)
    # Create valid set from train_set which is 1% of train set
    X_train, X_valid, counts_train, counts_valid, smi_train, smi_valid, y_train, y_valid = train_test_split(X_train, counts_train, smi_train, y_train, train_size=0.99, test_size=0.01, random_state=42)
    # Create a dataset object for each set
    train_dataset = FGDataset(X_train, counts_train, y_train, smi_train)
    valid_dataset = FGDataset(X_valid, counts_valid, y_valid, smi_valid)
    test_dataset = FGDataset(X_test, counts_test, y_test, smi_test)

    # Create a dataloader object for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, valid_loader, test_loader