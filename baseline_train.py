import os
import wandb
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Chemprop imports
from habnet.featuriser.featurise import Featuriser, MOL_TYPES
from habnet.featuriser.habnet_featurizer import AtomHAbNetFeaturizer
from habnet.utilities.utilities import evaluate_predictions_combined, extract_true_targets
from chemprop import data, featurizers, nn
from chemprop.nn import metrics
from chemprop.models import multi

# -------------------------------
# Step 1: Initialize Weights & Biases
# -------------------------------
wandb.init(project="chemprop_hyperparam_tuning", config={
    "batch_size": 64,
    "epochs": 100,
    "depth": 3,
    # "ffn_num_layers": 3,
    "dropout": 0.2,
    # "message_hidden_dim": 256,
    # "ffn_hidden_dim": 256,
    # "max_lr": 0.001,
    # "init_lr": 0.0001,
    # "final_lr": 0.0001,
    # "warmup_epochs": 5,
    # "activation": "relu",
    # "aggregation": "mean",
    # "batch_norm": False,
    # "bond_message_passing_shared": False
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
        nn.BondMessagePassing(depth=config.depth, 
                              d_v=featurizer.atom_fdim, dropout=config.dropout)
        for _ in range(len(MOL_TYPES))
    ],
    n_components=len(MOL_TYPES),
    shared= False#config.bond_message_passing_shared
)

# agg = getattr(nn, f"{config.aggregation.capitalize()}Aggregation")()  # Dynamically select aggregation type
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

ffn = nn.RegressionFFN(
    input_dim=mcmp.output_dim,
    hidden_dim=300,
    n_tasks=5,
    dropout=0.2,
    output_transform=output_transform,
    # n_layers=config.ffn_num_layers,
    # activation=config.activation,
    # criterion=metrics.RMSE(task_weights=[1, 1, 1, 100, 100])
)

# metric_list = [metrics.RMSE(task_weights=[1, 1, 1, 100, 100]), metrics.MAE(task_weights=[1, 1, 1, 100, 100]), metrics.R2Score(task_weights=[1, 1, 1, 100, 100])]  # First metric used for early stopping.
metric_list = [metrics.RMSE(), metrics.MAE(), metrics.R2Score()]
# Define full model
mcmpnn = multi.MulticomponentMPNN( mcmp, agg, ffn,
    metrics=metric_list, 
    batch_norm=False,
    init_lr=1e-4, max_lr=1e-3, final_lr=1e-4
)

# -------------------------------
# Step 4: Define Callbacks & Trainer
# -------------------------------
wandb_logger = WandbLogger(project="chemprop_experiments", name="baseline_run")

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename="mcmpnn-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3
)

# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     patience=10,
#     mode="min"
# )

trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],#, early_stopping],
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
# Step 5a: Prediction Evaluation
# -------------------------------

# Unscale val 
val_mcdset = data.MulticomponentDataset(val_datasets)
val_loader = data.build_dataloader(val_mcdset, shuffle=False, batch_size=config.batch_size)
val = trainer.test(mcmpnn, val_loader)
test = trainer.test(mcmpnn, test_loader)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger = False,
        enable_checkpointing = False,
        enable_progress_bar = False,
        accelerator = "auto",
        devices = 1,
        max_epochs = 100
    )
    predictions_test = trainer.predict(mcmpnn, test_loader)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger = False,
        enable_checkpointing = False,
        enable_progress_bar = False,
        accelerator = "auto",
        devices = 1,
        max_epochs = 100
    )
    predictions_val = trainer.predict(mcmpnn, val_loader)

preds_val_np = np.concatenate(predictions_val, axis=0)
preds_test_np = np.concatenate(predictions_test, axis=0)


# Get validation and test true labels
true_val = extract_true_targets(val_loader)
true_test = extract_true_targets(test_loader)

per_target_val_metrics = {
}
per_target_test_metrics = {
}

for i in range(true_val.shape[1]):
    rmse = np.sqrt(mean_squared_error(true_val[:, i], preds_val_np[:, i]))
    mae = mean_absolute_error(true_val[:, i], preds_val_np[:, i])
    r2 = r2_score(true_val[:, i], preds_val_np[:, i])
    per_target_val_metrics[f"val_rmse_{i}"] = rmse
    per_target_val_metrics[f"val_mae_{i}"] = mae
    per_target_val_metrics[f"val_r2_{i}"] = r2

for i in range(true_test.shape[1]):
    rmse = np.sqrt(mean_squared_error(true_test[:, i], preds_test_np[:, i]))
    mae = mean_absolute_error(true_test[:, i], preds_test_np[:, i])
    r2 = r2_score(true_test[:, i], preds_test_np[:, i])
    per_target_test_metrics[f"test_rmse_{i}"] = rmse
    per_target_test_metrics[f"test_mae_{i}"] = mae
    per_target_test_metrics[f"test_r2_{i}"] = r2



# -------------------------------
# Step 6: Log Final Metrics
# -------------------------------
test_results = trainer.test(mcmpnn, test_loader)[0]  # Get the first dictionary
wandb.log({
    "final_test_rmse": test_results.get("test/rmse", None),
    "final_test_mae": test_results.get("test/mae", None),
    "final_test_r2": test_results.get("test/r2", None),
    **per_target_val_metrics,
    **per_target_test_metrics

})

# -------------------------------
# Step 7: Plot Predictions and Residuals
# -------------------------------
evaluate_predictions_combined(preds_val_np, true_val, dataset_name="Validation", wandb_name=wandb.run.name, wandb_project= wandb.run.project ,output_path = "/home/calvin/code/chemprop_original/results")
evaluate_predictions_combined(preds_test_np,true_test,  dataset_name="Test", wandb_name=wandb.run.name, wandb_project = wandb.run.project, output_path = "/home/calvin/code/chemprop_original/results")

wandb.finish()
