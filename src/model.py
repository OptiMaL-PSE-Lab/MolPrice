import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

@gin.configurable(denylist=["input_size", "count_size"])
class FgLSTM(nn.Module):
    def __init__(self, input_size, count_size, embedding_size, hidden_lstm, hidden1_nn, hidden2_nn, output_size, dropout,lstm_size):
        super(FgLSTM, self).__init__()
        self.hidden_lstm = hidden_lstm
        self.hidden1_nn = hidden1_nn
        self.hidden2_nn = hidden2_nn
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.c_embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=2*embedding_size, hidden_size=hidden_lstm, num_layers=lstm_size, dropout=dropout,batch_first=True)
        # Use smm NN to predict price instead of linear layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_lstm, hidden1_nn),
            nn.ReLU(),
            nn.Linear(hidden1_nn, hidden2_nn),
            nn.ReLU(),
            nn.Linear(hidden2_nn, output_size)
        )

    def forward(self, x, c):
        seq_lengths, perm_idx = self.obtain_seq_ordering(x)
        x = x[perm_idx]
        c = c[perm_idx]
        seq_lengths = seq_lengths.cpu() # Need to convert to cpu for pack_padded_sequence
        
        x_emb = self.embedding(x)
        c_emb = self.c_embedding(c)
        c_emb = c_emb * c.unsqueeze(2)
        x = torch.cat((x_emb, c_emb), dim=2)

        x = pack_padded_sequence(x, seq_lengths, batch_first=True)
        packed_output, (h_n, c_n) = self.lstm(x)
        # Use c_n as the input to the dense layers
        if self.lstm.num_layers > 1:
           c_n = c_n[-1]
        # Reshape c_n to (batch_size, hidden_lstm)
        c_n = c_n.view(-1, self.hidden_lstm)    
        x = self.fc(c_n)
        return x, perm_idx
    
    def obtain_seq_ordering(self, x):
        # Shape of x is (batch_size, seq_len) with padding of 0s
        seq_lengths = torch.sum(x != 0, dim=1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        return seq_lengths, perm_idx
            

@gin.configurable
class FgAttention(nn.Module):
    def __init__(self, input_size, count_size, embedding_size, hidden_lstm, hidden1_nn, hidden2_nn, output_size, dropout, lstm_size):
        super(FgAttention, self).__init__()
        self.hidden_lstm = hidden_lstm
        self.hidden1_nn = hidden1_nn
        self.hidden2_nn = hidden2_nn
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.c_embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=2*embedding_size, hidden_size=hidden_lstm, num_layers=lstm_size, batch_first=True, dropout=dropout,bidirectional=True)
        self.pool = nn.Linear(hidden_lstm*2, 1, bias=False)
        # Use smm NN to predict price instead of linear layers
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_lstm, hidden1_nn),
            nn.ReLU(),
            nn.Linear(hidden1_nn, hidden2_nn),
            nn.ReLU(),
            nn.Linear(hidden2_nn, output_size)
        )

    def forward(self, x, c):
        seq_lengths, perm_idx = self.obtain_seq_ordering(x)
        x = x[perm_idx]
        c = c[perm_idx]
        seq_lengths = seq_lengths.cpu() # Need to convert to cpu for pack_padded_sequence
        
        x_emb = self.embedding(x)
        c_emb = self.c_embedding(x)
        c_emb = c_emb * c.unsqueeze(2)
        x = torch.cat((x_emb, c_emb), dim=2)

        x = pack_padded_sequence(x, seq_lengths, batch_first=True)
        packed_output, _ = self.lstm(x)
        padded_output, _ = pad_packed_sequence(packed_output, batch_first=True) # output of shape (batch_size, seq_len, hidden_lstm*2)
        # use self.pool on padded_output 
        attention_weights = self.pool(padded_output) # (batch_size, seq_len,1)
        # Replace 0s with -1e9 to avoid softmax giving 0
        attention_weights = torch.where(attention_weights == 0, torch.tensor(-1e9), attention_weights)
        attention_weights = nn.functional.softmax(attention_weights, dim=1) # (batch_size, seq_len)
        # Remove seq_len dimension by multiplying weights with padded_output and summing across seq_len
        context_vector = torch.sum(padded_output * attention_weights, dim=1) # (batch_size, hidden_lstm*2)
        x = self.fc(context_vector)
        return x, perm_idx
    
    def obtain_seq_ordering(self, x):
        # Shape of x is (batch_size, seq_len) with padding of 0s
        seq_lengths = torch.sum(x != 0, dim=1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        return seq_lengths, perm_idx


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, num_heads, hidden_size, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(embedding_size, num_heads, hidden_size) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        
        mask = self.create_mask(x)
        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded, mask)
        
        # Aggregate information across fragments using mean pooling
        pooled = embedded.mean(dim=1)

        # Pass through a fully connected layer with activation function
        output = self.fc(pooled)
        output = F.relu(output)

        # Squeeze to obtain shape (N_batch, 1)
        output = output.squeeze(dim=1)

        # Pass through a fully connected layer
        return output
    
    def create_mask(self, x):
        mask = torch.gt(x, 0)
        return mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attended, _ = self.multihead_attention(x, x, x, key_padding_mask=~mask)  # Invert the mask
        x = x + self.dropout(attended)
        x = self.layer_norm1(x)

        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.layer_norm2(x)

        return x

@gin.configurable  
class EarlyStopping:
    def __init__(self, patience, min_delta, deactivate=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_loss = 0
        self.deactivate = deactivate

    def early_stop(self, validation_loss):
        if validation_loss > self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False