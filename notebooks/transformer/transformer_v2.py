#!/usr/bin/env python
# coding: utf-8

# In[1]:


# basedir = "../.."
basedir = "."


# In[10]:

import os
from typing import List
import numpy as np
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from aimanager.transformer.dataset import MyDataset
from aimanager.transformer.model import GPTLikeTransformer, EarlyStopping
from pydantic import BaseModel


# In[8]:


class Config(BaseModel):
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 16
    patience: int = 20
    n_epochs: int = 2
    dataset_folder: str = "data/transformer_clone/trail_rounds_2_v2"
    pretrained_run_id: str = None


#     # experiment_names = ["trail_rounds_2"]
# experiment_names = ["random_1"]


# change string to compare os.environ with to enable ("enabled") or disable wandb
WANDB_ENABLED = os.environ.get("WANDB_MODE", "enabled") == "enabled"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"WANDB_ENABLED: {WANDB_ENABLED}")

if WANDB_ENABLED:
    wandb.init(project="algorithmic-institutions", entity="chm-hci")
    config = Config(**wandb.config)
    run_dir = f"temp/{wandb.run.id}"
else:
    config = Config()
    run_dir = "temp/transfomer_v2"


if config.pretrained_run_id is not None:
    prev_run_id = config.pretrained_run_id
    prev_run_path = f"chm-hci/algorithmic-institutions/{prev_run_id}"
    prev_run = wandb.Api().run(prev_run_path)
    prev_model_file = f"temp/{prev_run_id}/checkpoint.pt"
    prev_run.file(prev_model_file).download(replace=True)
    if WANDB_ENABLED:
        config = Config(**{**prev_run.config, **wandb.config})
    else:
        config = Config(**prev_run.config)


os.makedirs(run_dir, exist_ok=True)


# In[11]:

# Define hyperparameters
n_epochs = config.n_epochs
pad_idx = -1

folder = os.path.join(basedir, config.dataset_folder)
train_file = os.path.join(folder, "train_dataset.pt")
val_file = os.path.join(folder, "val_dataset.pt")
train_dataset = torch.load(train_file)
val_dataset = torch.load(val_file)

query_shape = train_dataset.queries.shape[-1]
input_shape = train_dataset.data.shape[-1]
num_classes = (train_dataset.targets).max().item() + 1


# In[11]:


# Create your DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize the model
model = GPTLikeTransformer(
    config.d_model,
    config.nhead,
    config.num_layers,
    config.dim_feedforward,
    num_classes,
    config.dropout,
    input_shape,
    query_shape,
).to(device)

if config.pretrained_run_id is not None:
    model.load_state_dict(torch.load(prev_model_file, map_location=device))

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx
)  # pad_idx is the index used for padding in your dataset
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)

# Initialize the early_stopping object
early_stopping = EarlyStopping(patience=config.patience, verbose=True, run_dir=run_dir)

for epoch in range(1, n_epochs + 1):
    # Training
    model.train()
    train_losses = []
    train_c_losses = []
    train_p_losses = []
    for batch in train_dataloader:
        x, y, q, mask, c_mask, p_mask = (t.to(device) for t in batch)

        optimizer.zero_grad()
        output = model(x, q)  # Shape: (batch_size, seq_len, num_classes)

        output = output.view(
            -1, output.size(-1)
        )  # Shape: (batch_size*seq_len, num_classes)
        y = y.contiguous().view(-1)  # Shape: (batch_size*seq_len,)

        mask = mask.contiguous().view(-1)  # Shape: (batch_size*seq_len,)
        c_mask = c_mask.contiguous().view(-1)  # Shape: (batch_size*seq_len,)
        p_mask = p_mask.contiguous().view(-1)  # Shape: (batch_size*seq_len,)

        loss = criterion(output, y)  # Compute the loss

        # Apply mask to the loss
        loss = loss * mask.float()
        c_loss = loss * c_mask.float()
        p_loss = loss * p_mask.float()

        # Normalize the loss
        loss = loss.sum() / mask.float().sum()
        c_loss = c_loss.sum() / c_mask.float().sum()
        p_loss = p_loss.sum() / p_mask.float().sum()

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_c_losses.append(c_loss.item())
        train_p_losses.append(p_loss.item())

    # Validation
    model.eval()
    val_losses = []
    val_c_losses = []
    val_p_losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            x, y, q, mask, c_mask, p_mask = (t.to(device) for t in batch)

            output = model(x, q)  # Shape: (batch_size, seq_len, num_classes)

            output = output.view(
                -1, output.size(-1)
            )  # Shape: (batch_size*seq_len, num_classes)
            y = y.contiguous().view(-1)  # Shape: (batch_size*seq_len,)
            mask = mask.contiguous().view(-1)  # Shape: (batch_size*seq_len,)
            c_mask = c_mask.contiguous().view(-1)  # Shape: (batch_size*seq_len,)
            p_mask = p_mask.contiguous().view(-1)  # Shape: (batch_size*seq_len,)

            loss = criterion(output, y)  # Compute the loss

            # Apply mask to the loss
            loss = loss * mask.float()
            c_loss = loss * c_mask.float()
            p_loss = loss * p_mask.float()

            # Normalize the loss
            loss = loss.sum() / mask.float().sum()
            c_loss = c_loss.sum() / c_mask.float().sum()
            p_loss = p_loss.sum() / p_mask.float().sum()

            val_losses.append(loss.item())
            val_c_losses.append(c_loss.item())
            val_p_losses.append(p_loss.item())

    # Print losses
    print(
        f"Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}"
    )
    if WANDB_ENABLED:
        wandb.log(
            {
                "Train Loss": np.mean(train_losses),
                "Val Loss": np.mean(val_losses),
                "Train Contribution Loss": np.mean(train_c_losses),
                "Val Contribution Loss": np.mean(val_c_losses),
                "Train Punishment Loss": np.mean(train_p_losses),
                "Val Punishment Loss": np.mean(val_p_losses),
                "Epoch": epoch,
            }
        )

    early_stopping(np.mean(val_losses), model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

if WANDB_ENABLED:
    wandb.save(early_stopping.get_checkpoint_file())


if WANDB_ENABLED:
    wandb.finish()

# %%
