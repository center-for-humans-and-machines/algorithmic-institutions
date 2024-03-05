import os
import torch
import torch.nn as nn
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, run_dir=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.run_dir = run_dir

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        file_name = os.path.join(self.run_dir, "checkpoint.pt")
        torch.save(model.state_dict(), file_name)  # <-- saves model
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        file_name = os.path.join(self.run_dir, "checkpoint.pt")
        model.load_state_dict(torch.load(file_name))

    def get_checkpoint_file(self):
        return os.path.join(self.run_dir, "checkpoint.pt")


# Define the Transformer model
class GPTLikeTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        num_classes,
        dropout,
        input_shape,
        query_shape,
    ):
        super(GPTLikeTransformer, self).__init__()

        # Define model dimensions
        self.d_model = d_model  # Number of features in input

        # Define the embedding layer that maps the input to the desired number of features
        self.embedding = nn.Linear(
            input_shape, d_model
        )  # Input: (batch_size, seq_len, input_shape), Output: (batch_size, seq_len, d_model)

        # Define the query embedding layer that maps the query input to the desired number of features
        self.query_embedding = nn.Linear(
            query_shape, d_model
        )  # Input: (batch_size, seq_len, query_shape), Output: (batch_size, seq_len, d_model)

        # Define the Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout=dropout
            ),
            num_layers,
        )  # Input: (seq_len, batch_size, d_model), Output: (seq_len, batch_size, d_model)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

        # Define the decoder layer that maps the output of the Transformer to the desired number of classes
        self.decoder = nn.Linear(
            d_model, num_classes
        )  # Input: (batch_size, seq_len, d_model), Output: (batch_size, seq_len, num_classes)

    # Function to generate a mask for the decoder to prevent it from "seeing" future tokens
    def _generate_square_subsequent_mask(self, sz):
        # We set the diagonal and above to -inf and the below diagonal to 0
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask  # Output: (seq_len, seq_len)

    def forward(self, x, query):
        # Embed the input
        x = self.embedding(x) * np.sqrt(
            self.d_model
        )  # Input: (batch_size, seq_len, input_shape), Output: (batch_size, seq_len, d_model)

        # Embed the query
        query = self.query_embedding(query) * np.sqrt(
            self.d_model
        )  # Input: (batch_size, seq_len, query_shape), Output: (batch_size, seq_len, d_model)

        # Transformer expects seq_len, batch, embedding_dim so we permute the dimensions
        x = x.permute(1, 0, 2)  # Output: (seq_len, batch_size, d_model)
        query = query.permute(1, 0, 2)  # Output: (seq_len, batch_size, d_model)

        # Create a mask for the Transformer
        tgt_mask = self._generate_square_subsequent_mask(query.size(0)).to(
            query.device
        )  # Output: (seq_len, seq_len)

        # Apply the Transformer
        x = self.transformer_decoder(
            tgt=query, memory=x, tgt_mask=tgt_mask, memory_mask=tgt_mask
        )  # Output: (seq_len, batch_size, d_model)

        # Apply dropout
        x = self.dropout(x)

        # Permute back to batch, seq_len, embedding_dim for linear layer
        x = x.permute(1, 0, 2)  # Output: (batch_size, seq_len, d_model)

        # Apply the decoder
        x = self.decoder(x)  # Output: (batch_size, seq_len, num_classes)

        return x  # No need to squeeze the last dimension, Output: (batch_size, seq_len, num_classes)
