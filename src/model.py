import gin
import math
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import r2_score

# User warning for key_padding mask as shown in commit fc94c90
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.modules.activation"
)


class CustomModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def mse_loss(
        self, logits: torch.FloatTensor, labels: torch.FloatTensor
    ) -> torch.Tensor:
        return F.mse_loss(logits, labels)

    def r2_score(self, logits: np.ndarray, labels: np.ndarray):
        return r2_score(logits, labels)


# class to be used for EFG/IFG model
@gin.configurable("LSTM")  # type:ignore
class FgLSTM(CustomModule):
    """
    FgLSTM is a PyTorch Lightning module that implements a LSTM-based model for price prediction given functional groups as inputs.

    Attributes:
        lr (float): The learning rate.
        hidden_lstm (int): The number of hidden units in the LSTM layer.
        hidden1_nn (int): The number of hidden units in the first fully connected layer.
        hidden2_nn (int): The number of hidden units in the second fully connected layer.
        lstm_size (float): The number of LSTM layers.
        embedding (nn.Embedding): The embedding layer for input sequences.
        c_embedding (nn.Embedding): The embedding layer for count sequences.
        lstm (nn.LSTM): The LSTM layer.
        pool (nn.Linear): The linear layer for pooling.
        fc (nn.Sequential): The fully connected layers for prediction..
    """

    def __init__(
        self,
        input_size: int,
        count_size: int,
        embedding_size: int,
        hidden_lstm: int,
        hidden1_nn: int,
        hidden2_nn: int,
        output_size: int,
        dropout: float,
        lstm_size: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="input_size, count_size")
        self.hidden_lstm = hidden_lstm
        self.hidden1_nn = hidden1_nn
        self.hidden2_nn = hidden2_nn
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=2 * embedding_size,
            hidden_size=hidden_lstm,
            num_layers=lstm_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.pool = nn.Linear(hidden_lstm * 2, 1, bias=False)
        # Use smm NN to predict price instead of linear layers
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_lstm, hidden1_nn),
            nn.ReLU(),
            nn.Linear(hidden1_nn, hidden2_nn),
            nn.ReLU(),
            nn.Linear(hidden2_nn, output_size),
        )

    def forward(self, x, c):
        seq_lengths, perm_idx = self.obtain_seq_ordering(x)
        x = x[perm_idx]
        c = c[perm_idx]
        seq_lengths = (
            seq_lengths.cpu()
        )  # Need to convert to cpu for pack_padded_sequence

        x_emb = self.embedding(x)
        # multiply x_emb with c to get the count embedding
        c = c.unsqueeze(2)
        c_emb = c * x_emb
        x = torch.cat((x_emb, c_emb), dim=2)

        x = pack_padded_sequence(x, seq_lengths, batch_first=True)
        packed_output, _ = self.lstm(x)
        padded_output, _ = pad_packed_sequence(
            packed_output, batch_first=True
        )  # output of shape (batch_size, seq_len, hidden_lstm*2)
        # use self.pool on padded_output
        attention_weights = self.pool(padded_output)  # (batch_size, seq_len,1)
        # Replace 0s with -1e9 to avoid softmax giving 0
        attention_weights = torch.where(
            attention_weights == 0, torch.tensor(-1e9), attention_weights
        )
        attention_weights = nn.functional.softmax(
            attention_weights, dim=1
        )  # (batch_size, seq_len)
        # Remove seq_len dimension by multiplying weights with padded_output and summing across seq_len
        context_vector = torch.sum(
            padded_output * attention_weights, dim=1
        )  # (batch_size, hidden_lstm*2)
        x = self.fc(context_vector)
        return x, perm_idx

    def obtain_seq_ordering(self, x):
        # Shape of x is (batch_size, seq_len) with padding of 0s
        seq_lengths = torch.sum(x != 0, dim=1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        return seq_lengths, perm_idx

    @gin.configurable(module="LSTM")  # type: ignore
    def configure_optimizers(
        self, optimizer: torch.optim.Optimizer
    ) -> OptimizerLRScheduler:
        opt = optimizer(self.parameters())  # type: ignore
        return opt

    def training_step(self, batch, batch_idx):
        inputs, counts, labels = batch["X"], batch["c"], batch["y"]
        # use model to get output
        output, perm_idx = self.forward(inputs, counts)
        labels = labels[perm_idx]
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, counts, labels = batch["X"], batch["c"], batch["y"]
        output, perm_idx = self.forward(inputs, counts)
        labels = labels[perm_idx]
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        r2_score = self.r2_score(output.cpu().numpy(), labels.cpu().numpy())
        r2_score = torch.tensor(r2_score)
        scores_to_log = {"val_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return loss, r2_score

    def test_step(self, batch, batch_idx):
        inputs, counts, labels = batch["X"], batch["c"], batch["y"]
        output, perm_idx = self.forward(inputs, counts)
        labels = labels[perm_idx]
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        r2_score = self.r2_score(output.cpu().numpy(), labels.cpu().numpy())
        r2_score = torch.tensor(r2_score)
        scores_to_log = {"test_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return loss, r2_score


# Used for Fingerprints
@gin.configurable("FP")  # type:ignore
class Fingerprints(CustomModule):
    def __init__(
        self, input_size, hidden_size_1: int, hidden_size_2: int, hidden_size_3: int
    ):
        super(Fingerprints, self).__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, 1),
        )
        self.save_hyperparameters()

    def forward(self, x):
        x = self.neural_network(x)
        return x

    @gin.configurable(module="FP")  # type: ignore
    def configure_optimizers(
        self, optimizer: torch.optim.Optimizer
    ) -> OptimizerLRScheduler:
        opt = optimizer(self.parameters())  # type: ignore
        return opt

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        r2_score = self.r2_score(output.cpu().numpy(), labels.cpu().numpy())
        r2_score = torch.tensor(r2_score)
        scores_to_log = {"val_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        r2_score = self.r2_score(output.cpu().numpy(), labels.cpu().numpy())
        r2_score = torch.tensor(r2_score)
        scores_to_log = {"test_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return loss


# class to be used for SMILES model
@gin.configurable("Transformer")  # type:ignore
class TransformerEncoder(CustomModule):
    def __init__(
        self,
        input_size,
        embedding_size,
        num_heads,
        hidden_size,
        num_layers,
        dropout,
    ):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embedding_size, 1)
        self.save_hyperparameters(ignore="input_size")

    def forward(self, x):
        # x has dimensions of (N_batch, N_seq)
        embedded = self.embedding(
            x
        )  # embedding has dimensions of (N_batch, N_seq, embedding_size)
        mask = self.create_mask(x)
        embedded = self.positional_encoding(embedded)
        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded, mask)

        # Aggregate information across fragments using mean pooling
        pooled = embedded.mean(dim=1)
        # Pass through a fully connected layer with activation function
        output = self.fc(pooled)
        output = F.relu(output)
        return output

    def create_mask(self, x):
        #! Assumes that masking token is 0 i.e. padding token
        mask = torch.gt(x, 0)
        # convert mask tensor to boolean list not of type Tensor, but type bool
        mask = mask.bool()
        return mask

    @gin.configurable(module="Transformer")  # type: ignore
    def configure_optimizers(
        self, optimizer: torch.optim.Optimizer, decay_rate: float, warmup_epochs: int
    ) -> OptimizerLRScheduler:

        opt = optimizer(self.parameters())  # type: ignore

        # Define the learning rate schedule
        def lr_schedule(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return decay_rate ** (epoch - warmup_epochs)

        scheduler = LambdaLR(opt, lr_schedule)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # The scheduler updates the learning rate after each epoch
            },
        }

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        # use model to get output
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        # convert output to floatTensor
        loss = self.mse_loss(output, labels)  # type: ignore
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)  # type: ignore
        r2_score = self.r2_score(output.cpu().numpy(), labels.cpu().numpy())
        r2_score = torch.tensor(r2_score)
        scores_to_log = {"val_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)
        return loss, r2_score

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)  # type: ignore
        r2_score = self.r2_score(output.cpu().numpy(), labels.cpu().numpy())
        r2_score = torch.tensor(r2_score)
        scores_to_log = {"test_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return loss


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, embedding_size: int, num_heads: int, hidden_size: int, dropout: float
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=num_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask) -> torch.FloatTensor:
        attended, _ = self.multihead_attention(
            x, x, x, key_padding_mask=~mask
        )  # Invert the mask
        x = x + self.dropout(attended)
        x = self.layer_norm1(x)

        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.layer_norm2(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe should have (1, max_len, d_model) shape as batch_first is True
        pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, max_length, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return self.dropout(x)
