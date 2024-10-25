import gin
import math
import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics import R2Score
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    SpearmanCorrCoef,
)
from transformers import RobertaConfig, RobertaModel, get_linear_schedule_with_warmup

# User warning for key_padding mask as shown in commit fc94c90
import warnings


warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.modules.activation"
)


class CustomModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.test_predictions = []
        self.test_labels = []

    def mse_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        metric = MeanSquaredError()
        metric = metric.to(device=logits.device)
        return metric(logits, labels)

    def mae_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        metric = MeanAbsoluteError()
        metric = metric.to(device=logits.device)
        return metric(logits, labels)

    def r2_score(self, logits: torch.Tensor, labels: torch.Tensor):
        metric = R2Score()
        metric = metric.to(device=logits.device)
        return metric(target=labels, preds=logits)

    def spearman_corr(self, logits: torch.Tensor, labels: torch.Tensor):
        metric = SpearmanCorrCoef()
        metric = metric.to(device=logits.device)
        return metric(logits, labels)

    def on_test_epoch_end(self):
        all_predictions = torch.cat(self.test_predictions)
        all_labels = torch.cat(self.test_labels)
        # Calculate R2 score on the entire dataset
        spearmean_corr = self.spearman_corr(all_predictions, all_labels.view(-1, 1))
        r2_score = self.r2_score(all_predictions, all_labels.view(-1, 1))
        self.log_dict({"rs": spearmean_corr, "r2_score": r2_score})

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> list[torch.Tensor]:
        if self == "FgLSTM":
            inputs, counts = batch["X"], batch["c"]
            return self(inputs, counts)
        else:
            inputs = batch["X"]
            return self(inputs)[0]


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
            "train_mse",
            loss,
            on_step=True,
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
        r2_score = self.r2_score(output, labels)
        if wandb.run:
            if self.trainer.global_step == 0:
                wandb.define_metric("val_loss", summary="min")
                wandb.define_metric("r2_score", summary="max")
        scores_to_log = {"val_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return loss, r2_score

    def test_step(self, batch, batch_idx):
        inputs, counts, labels = batch["X"], batch["c"], batch["y"]
        output, perm_idx = self.forward(inputs, counts)
        labels = labels[perm_idx]
        labels = labels.view(-1, 1)
        self.test_predictions.append(output)
        self.test_labels.append(labels)
        mse_loss = self.mse_loss(output, labels)
        mae_loss = self.mae_loss(output, labels)
        scores_to_log = {"test_mse": mse_loss, "test_mae": mae_loss}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return mse_loss


# Used for Fingerprints
@gin.configurable("FP")  # type:ignore
class Fingerprints(CustomModule):
    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        hidden_size_3: int,
        dropout: float,
        loss_hp: float,
        loss_sep: bool,
        two_d: bool,  # whether dataloader included 2D info in fingerprint
    ):

        super(Fingerprints, self).__init__()
        if two_d:
            input_size += 10

        self.neural_network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_3, 10),
        )
        self.linear = nn.Linear(10, 1)
        self.loss_hp = loss_hp
        self.loss_sep = loss_sep
        self.dropout = dropout
        self.input_size = input_size

    def forward(self, x):
        x = self.neural_network(x)
        x = self.linear(x)
        return x, x

    def forward_sep(self, x):
        z = self.neural_network(x)
        z = F.leaky_relu(z, 0.1)
        x = F.dropout(z, 0.2)
        x = self.linear(x)
        return x, z

    @gin.configurable(module="FP")  # type: ignore
    def configure_optimizers(
        self, optimizer: torch.optim.Optimizer
    ) -> OptimizerLRScheduler:
        opt = optimizer(self.parameters())  # type: ignore
        return opt

    def contrastive_loss(self, z, out, labels):
        z_es, z_hs = z.chunk(2, dim=0)
        es_label, _ = labels.chunk(2, dim=0)
        es_out, hs_out = out.chunk(2, dim=0)
        cont_loss = self.cosine_similarity_loss(z_hs, z_es, 0)
        mse_loss = self.mse_loss(es_out, es_label)
        total_loss = self.loss_hp * mse_loss + (1 - self.loss_hp) * cont_loss
        return total_loss, mse_loss, cont_loss

    def cosine_similarity_loss(self, h_HS, h_ES, lambda_val):
        # Normalize the embeddings to have unit norm (for cosine similarity)
        h_HS = F.normalize(h_HS, p=2, dim=1)  # Shape: (B_HS, N)
        h_ES = F.normalize(h_ES, p=2, dim=1)  # Shape: (B_ES, N)

        # Intra-class cosine similarity (HS-to-HS and ES-to-ES)
        cosine_sim_HS = torch.mm(h_HS, h_HS.t())  # Shape: (B_HS, B_HS)
        cosine_sim_ES = torch.mm(h_ES, h_ES.t())  # Shape: (B_ES, B_ES)
        # take absolute value
        cosine_sim_HS = torch.abs(cosine_sim_HS)
        cosine_sim_ES = torch.abs(cosine_sim_ES)

        # Avoid self-similarity in the intra-class loss (mask the diagonal)
        mask_HS = torch.eye(h_HS.size(0), device=h_HS.device).bool()
        mask_ES = torch.eye(h_ES.size(0), device=h_ES.device).bool()

        intra_loss_HS = (
            1 - cosine_sim_HS.masked_select(~mask_HS).mean()
        )  # Maximize cosine similarity for HS
        intra_loss_ES = (
            1 - cosine_sim_ES.masked_select(~mask_ES).mean()
        )  # Maximize cosine similarity for ES

        # Inter-class cosine similarity (HS-to-ES)
        cosine_sim_inter = torch.mm(h_HS, h_ES.t())  # Shape: (B_HS, B_ES)
        cosine_sim_inter = 1 + cosine_sim_inter

        # Minimize inter-class cosine similarity
        inter_loss = (
            cosine_sim_inter.mean()
        )  # Minimize cosine similarity between HS and ES

        # Total loss
        total_loss = (
            intra_loss_HS + intra_loss_ES + 2 * inter_loss
        )  # + (entropy_loss_hs + entropy_loss_es)
        return total_loss

    def training_step(self, batch, batch_idx):
        labels = batch["y"]
        inputs = batch["X"]
        labels = labels.view(-1, 1)
        if self.loss_sep:
            output, z = self.forward_sep(inputs)
            total_loss, mse_loss, cont_loss = self.contrastive_loss(z, output, labels)
            self.log_dict(
                {
                    "train_total": total_loss,
                    "train_mse": mse_loss,
                    "train_contrastive": cont_loss,
                },
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            output, _ = self.forward(inputs)
            mse_loss = self.mse_loss(output, labels)
            total_loss = mse_loss
            self.log(
                "train_mse",
                mse_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return total_loss

    def validation_step(self, batch, batch_idx):
        labels = batch["y"]
        inputs = batch["X"]
        labels = labels.view(-1, 1)
        if self.loss_sep:
            output, z = self.forward_sep(inputs)
            total_loss, mse_loss, cont_loss = self.contrastive_loss(z, output, labels)
            self.log_dict(
                {"val_mse": mse_loss, "val_contrastive": cont_loss},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            output, _ = output.chunk(2, dim=0)
            labels, _ = labels.chunk(2, dim=0)
        else:
            inputs = batch["X"]
            output, _ = self.forward(inputs)
            total_loss = self.mse_loss(output, labels)

        r2_score = self.r2_score(output, labels)
        if wandb.run:
            if self.trainer.global_step == 0:
                wandb.define_metric("val_mse", summary="min")
                wandb.define_metric("r2_score", summary="max")

        scores_to_log = {"val_loss": total_loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return total_loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        labels = labels.view(-1, 1)
        if self.loss_sep:
            output, z = self.forward_sep(inputs)
            total_loss, mse_loss, contrastive_loss = self.contrastive_loss(
                z, output, labels
            )
            # only log cont loss here
            self.log(
                "test_contrastive",
                contrastive_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            output, _ = self.forward(inputs)
            mse_loss = self.mse_loss(output, labels)
            total_loss = mse_loss
        self.test_predictions.append(output)
        self.test_labels.append(labels)
        mae_loss = self.mae_loss(output, labels)
        scores_to_log = {"test_mse": mse_loss, "test_mae": mae_loss}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return total_loss


@gin.configurable("RoBERTa")  # type:ignore
class RoBERTaClassification(CustomModule):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
    ):
        super(RoBERTaClassification, self).__init__()
        self.config = RobertaConfig.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
        self.pretrained_model = RobertaModel.from_pretrained(
            "DeepChem/ChemBERTa-10M-MLM", config=self.config
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out = self.pretrained_model(x)
        out = out.pooler_output
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        loss = self.mse_loss(output, labels)
        self.log(
            "train_mse",
            loss,
            on_step=True,
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
        r2_score = self.r2_score(output, labels)
        if wandb.run:
            if self.trainer.global_step == 0:
                wandb.define_metric("val_mse", summary="min")
                wandb.define_metric("r2_score", summary="max")
        scores_to_log = {"val_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        self.test_predictions.append(output)
        self.test_labels.append(labels)
        mse_loss = self.mse_loss(output, labels)
        mae_loss = self.mae_loss(output, labels)
        scores_to_log = {"test_mse": mse_loss, "test_mae": mae_loss}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)
        return mse_loss

    @gin.configurable(module="RoBERTa")  # type: ignore
    def configure_optimizers(self, optimizer: torch.optim.Optimizer, ):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-4,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        opt = optimizer(optimizer_grouped_parameters)  # type: ignore
        return opt


# class to be used for SMILES model
@gin.configurable("Transformer")  # type:ignore
class TransformerEncoder(CustomModule):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(TransformerEncoder, self).__init__()
        self.model_dim = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embedding_size, 1)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x) -> torch.Tensor:
        embedded = self.embedding(
            x
        )  # embedding has dimensions of (N_batch, N_seq, embedding_size)
        # scale the embedding by sqrt of embedding size
        embedded = embedded * math.sqrt(self.model_dim)
        mask = self.create_mask(x)
        embedded = self.positional_encoding(embedded)
        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded, mask)

        cls_hidden_state = embedded.max(dim=1)[0]  # (N_batch, embedding_size)
        output = self.fc(cls_hidden_state)
        output = F.relu(output)
        return output

    def create_mask(self, x):
        # * Assumes that masking token is 0 i.e. padding token
        mask = torch.gt(x, 0)
        # convert mask tensor to boolean list not of type Tensor, but type bool
        mask = mask.bool()
        return mask

    @gin.configurable(module="Transformer")  # type: ignore
    def configure_optimizers(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> OptimizerLRScheduler:

        opt = optimizer(self.parameters())  # type: ignore
        scheduler = scheduler(opt, max_lr=gin.REQUIRED, total_steps=gin.REQUIRED)  # type: ignore

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,  # scheduler is OneCycleLR from gin file
                "interval": "step",  # The scheduler updates the learning rate after each epoch
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
            "train_mse",
            loss,
            on_step=True,
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
        r2_score = self.r2_score(output, labels)
        if wandb.run:
            if self.trainer.global_step == 0:
                wandb.define_metric("val_mse", summary="min")
                wandb.define_metric("r2_score", summary="max")
        scores_to_log = {"val_loss": loss, "r2_score": r2_score}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["X"], batch["y"]
        output = self.forward(inputs)
        labels = labels.view(-1, 1)
        self.test_predictions.append(output)
        self.test_labels.append(labels)
        mse_loss = self.mse_loss(output, labels)
        mae_loss = self.mae_loss(output, labels)
        scores_to_log = {"test_mse": mse_loss, "test_mae": mae_loss}
        self.log_dict(scores_to_log, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore
        return mse_loss


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, embedding_size: int, num_heads: int, hidden_size: int, dropout: float
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
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
        x = x + attended
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
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, max_length, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return self.dropout(x)
