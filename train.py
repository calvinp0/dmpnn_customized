import os
import wandb
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Chemprop imports
from habnet.featuriser.featurise import Featuriser, MOL_TYPES
from habnet.featuriser.habnet_featurizer import AtomHAbNetFeaturizer
from chemprop import data, featurizers, nn
from chemprop.nn import metrics
from chemprop.models import multi

# -------------------------------
# Step 1: Initialize Weights & Biases
# -------------------------------
wandb.init(project="chemprop_hyperparam_tuning", config={
    "batch_size": 64,
    "epochs": 100
})

# Get W&B configuration
config = wandb.config


# -------------------------------
# Step 2: Load and Preprocess Data
# -------------------------------
feat_data = Featuriser(
    os.path.expanduser("~/code/chemprop_phd_customised/habnet/data/processed/sdf_data"),
    path="/home/calvin/code/chemprop_phd_customised/habnet/data/processed/target_data/target_data.csv"
)

component_to_split_by = 0
mols = [d.mol for d in feat_data[component_to_split_by]]

train_indices, val_indices, test_indices = data.make_split_indices(mols, "kennard_stone", (0.8, 0.1, 0.1), num_replicates=10)
train_data, val_data, test_data = data.split_data_by_indices(feat_data, train_indices, val_indices, test_indices)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=AtomHAbNetFeaturizer())

train_datasets = [data.MoleculeDataset(train_data[0][i], featurizer) for i in range(len(MOL_TYPES))]
val_datasets = [data.MoleculeDataset(val_data[0][i], featurizer) for i in range(len(MOL_TYPES))]
test_datasets = [data.MoleculeDataset(test_data[0][i], featurizer) for i in range(len(MOL_TYPES))]

train_mcdset = data.MulticomponentDataset(train_datasets)
scaler = train_mcdset.normalize_targets()
val_mcdset = data.MulticomponentDataset(val_datasets)
val_mcdset.normalize_targets(scaler)
test_mcdset = data.MulticomponentDataset(test_datasets)

train_loader = data.build_dataloader(train_mcdset, batch_size=config.batch_size)
val_loader = data.build_dataloader(val_mcdset, shuffle=False, batch_size=config.batch_size)
test_loader = data.build_dataloader(test_mcdset, shuffle=False, batch_size=config.batch_size)

# -------------------------------
# Step 3: Define the Model
# -------------------------------
mcmp = nn.MulticomponentMessagePassing(
    blocks=[
        nn.BondMessagePassing(depth=config.depth, d_v=featurizer.atom_fdim, dropout=config.dropout)
        for _ in range(len(MOL_TYPES))
    ],
    n_components=len(MOL_TYPES),
    shared=config.bond_message_passing_shared
)

agg = getattr(nn, f"{config.aggregation.capitalize()}Aggregation")()  # Dynamically select aggregation type

output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

ffn = nn.RegressionFFN(
    input_dim=mcmp.output_dim,
    hidden_dim=config.ffn_hidden_dim,
    n_tasks=5,
    dropout=config.dropout,
    output_transform=output_transform,
    n_layers=config.ffn_num_layers,
    activation=config.activation
)

metric_list = [metrics.RMSE(), metrics.MAE(), metrics.R2Score()]  # First metric used for early stopping.

# Define full model
mcmpnn = multi.MulticomponentMPNN(
    mcmp, agg, ffn, metrics=metric_list, batch_norm=config.batch_norm,
    init_lr=config.init_lr, max_lr=config.max_lr, final_lr=config.final_lr
)

# -------------------------------
# Step 4: Define Callbacks & Trainer
# -------------------------------
wandb_logger = WandbLogger(project="chemprop_hyperparam_tuning")

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename="mcmpnn-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min"
)

trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stopping],
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=config.epochs
)

# -------------------------------
# Step 5: Train and Evaluate Model
# -------------------------------
trainer.fit(mcmpnn, train_loader, val_loader)
trainer.test(mcmpnn, test_loader)

# -------------------------------
# Step 6: Log Final Metrics
# -------------------------------
test_results = trainer.test(mcmpnn, test_loader)[0]  # Get the first dictionary
wandb.log({
    "final_test_rmse": test_results["test/rmse"],
    "final_test_mae": test_results["test/mae"],
    "final_test_r2": test_results["test/r2"]
})

wandb.finish()