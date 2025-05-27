import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import ray
from ray import tune
from ray.tune import Tuner
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
from ray.train.torch import TorchTrainer
from lightning import pytorch as pl
from chemprop import data, featurizers, nn
from chemprop.models import multi
from chemprop.nn.metrics import LossFunctionRegistry, ChempropMetric
from chemprop.nn.metrics import MSE
from chemprop.CUSTOM.featuriser.featurise import Featuriser, MOL_TYPES  # your custom featuriser
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

pl.seed_everything(42)

# 0) Paths
raw_csv   = '/home/calvin.p/Code/chemprop_original/DATA/target_data/kinetics_summary.csv'
out_csv   = '/home/calvin.p/Code/chemprop_original/DATA/target_data/temp_target_kinetic_data.csv'

# 1) Load
df = pd.read_csv(raw_csv)

#  1.a) Filter for label == 'k_for (TST+T)'
df = df[df['label'] == 'k_for (TST+T)'].copy()

# 2) Drop any rows missing the columns we care about
df = df.dropna(subset=['rxn', 'label', 'A', 'n', 'Ea'])

# 3) Dedupe by (rxn, label), keeping the first occurrence
df = df.drop_duplicates(subset=['rxn','label'], keep='first')


# 4) Filter out non-positive A (cannot log-transform those)
df = df[df['A'] > 0].copy()


# 5) Log10-transform A into a new column
df['A_log10'] = np.log10(df['A'])

# 6) Fit & apply Yeo–Johnson transform to the entire Ea column
# Fit on a pure numpy array
Ea_vals = df['Ea'].to_numpy().reshape(-1, 1)   # ndarray, no column names
pt_ea = PowerTransformer(method='yeo-johnson')
Ea_yj = pt_ea.fit_transform(Ea_vals).ravel()
df['Ea_yj'] = Ea_yj

# 7) (Optional) inspect your transforms
print("A_log10 summary:\n", df['A_log10'].describe())
print("Ea_yj   summary:\n", df['Ea_yj'].describe())

# 8) Save out
df.to_csv(out_csv, index=False)
print(f"Wrote processed targets to {out_csv}")
#######################################
feat_data = Featuriser(os.path.expanduser("~/code/chemprop_phd_customised/habnet/data/processed/sdf_data"), filter_rules={'label': 'k_for (TST+T)'},
                       path = out_csv, set_col_index=False, target_col=['A_log10','n', 'Ea_yj'],
                       include_extra_features = False)

component_to_split_by = 0
mols = [d.mol for d in feat_data[component_to_split_by]]

train_indices, val_indices, test_indices = data.make_split_indices(mols, "kennard_stone", (0.8, 0.1, 0.1))


train_data, val_data, test_data = data.split_data_by_indices(
    feat_data, train_indices, val_indices, test_indices
)

rxn_ids = []
for i in range(len(train_data[0][0])):
    rxn_name = train_data[0][0][i].name
    # Need to remove _r1h or _r2h from the reaction name
    rxn_name = rxn_name.replace('_r1h', '').replace('_r2h', '')
    rxn_ids.append(rxn_name)

# Read extra info
import pandas as pd
atom_extra_feats = pd.read_csv("/home/calvin.p/Code/chemprop_original/DATA/sdf_dataall_sdf_features.csv")
atom_extra_feats.columns

import numpy as np

def rbf_expand(values, num_centers=20, r_min=None, r_max=None, gamma=None):
    values = np.asarray(values)
    if r_min is None:
        r_min = float(np.min(values))
    if r_max is None:
        r_max = float(np.max(values))
    # Generate evenly spaced centers
    centers = np.linspace(r_min, r_max, num_centers)
    if gamma is None:
        # Set gamma so adjacent bases overlap well
        gamma = 1.0 / (centers[1] - centers[0])**2
    # Compute RBF
    expanded = np.exp(-gamma * (values[..., None] - centers)**2)
    return expanded  # shape: (len(values), num_centers)

def dihedral_to_sin_cos(dihedrals_deg):
    dihedrals_rad = np.deg2rad(dihedrals_deg)
    sin_vals = np.sin(dihedrals_rad)
    cos_vals = np.cos(dihedrals_rad)
    return np.stack([sin_vals, cos_vals], axis=-1)

def normalize_angle(angle_deg, a_min=0.0, a_max=180.0):
    # If your angles can go up to 180, otherwise adjust a_max as needed
    angle_deg = np.asarray(angle_deg)
    return (angle_deg - a_min) / (a_max - a_min)

import numpy as np
import pandas as pd

# Your DataFrame and rxn_ids
# atom_extra_feats: DataFrame with all data
# rxn_ids: list of rxn_ids in the train set

# 1. Subset for fitting
train_mask = atom_extra_feats['rxn_id'].isin(rxn_ids)
train_feats = atom_extra_feats[train_mask]

# 2. Compute parameters from training data
num_centers = 16
r_min = train_feats['radius'].min()
r_max = train_feats['radius'].max()
a_min = train_feats['angle'].min()
a_max = train_feats['angle'].max()

# 3. Apply to all data using train params
radius_rbf = rbf_expand(atom_extra_feats['radius'].values, num_centers=num_centers, r_min=r_min, r_max=r_max)
dihedral_sc = dihedral_to_sin_cos(atom_extra_feats['dihedral'].values)
angle_norm = normalize_angle(atom_extra_feats['angle'].values, a_min=a_min, a_max=a_max)

# 4. Stack or add to DataFrame
for i in range(num_centers):
    atom_extra_feats[f'radius_rbf_{i}'] = radius_rbf[:, i]
atom_extra_feats['dihedral_sin'] = dihedral_sc[:, 0]
atom_extra_feats['dihedral_cos'] = dihedral_sc[:, 1]
atom_extra_feats['angle_norm'] = angle_norm

def get_atom_feats_for_dp(dp_name, atom_extra_feats):
    rxn_id, mol_type = dp_name.rsplit('_', 1)
    subset = atom_extra_feats[
        (atom_extra_feats['rxn_id'] == rxn_id) &
        (atom_extra_feats['mol_type'] == mol_type)
    ]
    # Sort by focus_atom_idx
    subset = subset.sort_values("focus_atom_idx")
    return subset


def attach_morgan_to_dps(dps):
    new_dps = []
    drop_cols = ['rxn_id', 'mol_type', 'focus_atom_idx', 'path', 'radius', 'angle', 'dihedral','focus_atom_symbol']
    for dp in dps:
        # pick the mol you actually want to fingerprint:
        mol = dp.mol if not isinstance(dp.mol, tuple) else dp.mol[0]
        extra_atom_feats = get_atom_feats_for_dp(dp.name, atom_extra_feats).drop(columns=drop_cols).values
        # Make sure extra_atom_feats is not empty
        if extra_atom_feats.shape[0] == 0:
            print(f"No extra atom features found for datapoint {dp.name}. Check your atom_extra_feats DataFrame.")
            continue
        new_dp = data.MoleculeDatapoint(
            mol=dp.mol,
            y=dp.y,
            weight=dp.weight,
            gt_mask=dp.gt_mask,
            lt_mask=dp.lt_mask,
            V_f=extra_atom_feats,
            E_f=dp.E_f,
            V_d=dp.V_d,
            x_d=dp.x_d,      # <-- lowercase x_d
            x_phase=dp.x_phase,
            name=dp.name
        )
        new_dps.append(new_dp)
    return new_dps


featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=26)

train_datasets = [
    data.MoleculeDataset(
        attach_morgan_to_dps(train_data[0][i]),
        featurizer
    )
    for i in range(len(MOL_TYPES))
]

val_datasets = [
    data.MoleculeDataset(
        attach_morgan_to_dps(val_data[0][i]),
        featurizer
    )
    for i in range(len(MOL_TYPES))
]
test_datasets = [
    data.MoleculeDataset(
        attach_morgan_to_dps(test_data[0][i]),
        featurizer
    )
    for i in range(len(MOL_TYPES))
]


# file: my_losses.py
import torch
import torch.nn.functional as F
from chemprop.nn.metrics import ChempropMetric, LossFunctionRegistry

@LossFunctionRegistry.register("huber_mse")
class HuberMSE(ChempropMetric):
    """
    A mixed loss that applies:
      – Huber (smooth-L1) on the first task (log10(A))
      – MSE on the remaining tasks (n, Ea)
    and then weights each task via task_weights.
    """
    def __init__(self,
                 task_weights: list[float] = 1.0,
                 delta: float = 1.0):
        super().__init__(task_weights)
        self.delta = delta
        self.register_buffer("delta_buf", torch.tensor(delta))

    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, *args):
        # preds, targets: shape (B, 3)
        # 1) Huber on the 0th channel
        diff0    = preds[:, 0] - targets[:, 0]
        absdiff0 = diff0.abs()
        mask0    = absdiff0 <= self.delta_buf
        huber0   = torch.where(mask0,
                               0.5 * diff0.pow(2),
                               self.delta_buf * (absdiff0 - 0.5 * self.delta_buf))

        # 2) MSE on channels 1 and 2
        mse12    = (preds[:, 1:] - targets[:, 1:]).pow(2)

        # 3) concatenate back into (B,3)
        return torch.cat([huber0.unsqueeze(1), mse12], dim=1)



# 1. Prepare your data exactly as in your notebook:
#    - build train_mcdset, val_mcdset, scaler, featurizer, attach extra features, etc.

train_mcdset = data.MulticomponentDataset(train_datasets)
scaler = train_mcdset.normalize_targets()

val_mcdset = data.MulticomponentDataset(val_datasets)
val_mcdset.normalize_targets(scaler)
test_mcdset = data.MulticomponentDataset(test_datasets)
test_mcdset.normalize_targets(scaler)


train_loader = data.build_dataloader(train_mcdset, batch_size=128, shuffle=True, num_workers=14, pin_memory=True)
val_loader = data.build_dataloader(val_mcdset, shuffle=False, batch_size=64)
test_loader = data.build_dataloader(test_mcdset, shuffle=False, batch_size=64)
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

# -----------------------------------------------------------------------------
# 3) train_model: builds, fits, then evaluates & reports raw metrics
# -----------------------------------------------------------------------------
def train_model(config):
    # Unpack hyperparameters
    depth          = int(config["depth"])
    mp_hdim        = int(config["message_hidden_dim"])
    ffn_hdim       = int(config["ffn_hidden_dim"])
    ffn_layers     = int(config["ffn_num_layers"])
    dropout        = config["dropout"]
    warmup_epochs  = int(config["warmup_epochs"])
    init_lr        = config["init_lr"]
    max_lr         = config["max_lr"]
    final_lr       = config["final_lr"]
    batch_size     = int(config["batch_size"])
    max_epochs     = int(config["max_epochs"])

    # Build model
    mp_blocks = [
        nn.BondMessagePassing(depth=depth, dropout=dropout,
                              d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim,
                              d_h=mp_hdim)
        for _ in range(len(MOL_TYPES))
    ]
    mcmp = nn.MulticomponentMessagePassing(blocks=mp_blocks,
                                           n_components=len(MOL_TYPES),
                                           shared=False)
    agg  = nn.MeanAggregation()
    ffn  = nn.RegressionFFN(n_tasks=3,
                            input_dim=mcmp.output_dim,
                            hidden_dim=ffn_hdim,
                            n_layers=ffn_layers,
                            dropout=dropout,
                            criterion=HuberMSE([10.0,1.0,1.0], delta=0.1))
    model = multi.MulticomponentMPNN(message_passing=mcmp,
                                     agg=agg,
                                     predictor=ffn,
                                     metrics=[HuberMSE([10.0,1.0,1.0], delta=0.1)],
                                     warmup_epochs=warmup_epochs,
                                     init_lr=init_lr,
                                     max_lr=max_lr,
                                     final_lr=final_lr)

    # DataLoaders
    train_loader = data.build_dataloader(train_mcdset, batch_size=batch_size,
                                         shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = data.build_dataloader(val_mcdset, batch_size=batch_size,
                                         shuffle=False)

    # Ray‐wrapped Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()]
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_loader, val_loader)

    # ---- Now evaluate raw‐unit metrics on the test set ----
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_model = multi.MulticomponentMPNN.load_from_checkpoint(
        best_ckpt,
        message_passing=mcmp,
        agg=agg,
        predictor=ffn,
    ).eval()

    # collect scaled preds + truths
    scaled_preds = []
    scaled_trues = []
    for batch in data.build_dataloader(test_mcdset, batch_size=batch_size, shuffle=False):
        X, V, mask, Y = batch
        with torch.no_grad():
            p = best_model(X, V, mask).cpu().numpy()
        scaled_preds.append(p)
        scaled_trues.append(Y.cpu().numpy())
    scaled_preds = np.vstack(scaled_preds)
    scaled_trues = np.vstack(scaled_trues)

    # un‐scale back
    scaler = output_transform.to_standard_scaler()
    unscaled_preds = scaler.inverse_transform(scaled_preds)
    unscaled_trues = scaler.inverse_transform(scaled_trues)

    # raw A, n, Ea
    logA_p, n_p, Ea_yj_p = unscaled_preds.T
    logA_t, n_t, Ea_yj_t = unscaled_trues.T
    A_p     = 10**logA_p
    A_t     = 10**logA_t
    Ea_p    = pt_ea.inverse_transform(Ea_yj_p.reshape(-1,1)).ravel()
    Ea_t    = pt_ea.inverse_transform(Ea_yj_t.reshape(-1,1)).ravel()

    # clamp
    eps = 1e-8
    A_p  = np.clip(A_p, eps, None)
    Ea_p = np.clip(Ea_p, 0.0, None)

    # compute stats
    preds_final = np.vstack([A_p, n_p, Ea_p]).T
    trues_final = np.vstack([A_t, n_t, Ea_t]).T
    mae_vals  = mean_absolute_error(trues_final, preds_final, multioutput='raw_values')
    rmse_vals = root_mean_squared_error(trues_final, preds_final, multioutput='raw_values')
    r2_vals   = r2_score(trues_final, preds_final, multioutput='raw_values')
    r2_logA   = r2_score(logA_t, logA_p)
    pct_err_A = 100 * mae_vals[0] / np.mean(A_t)

    # report back to Tune
    tune.report(
        val_loss = trainer.callback_metrics["val_loss"].item(),
        raw_mae_A=mae_vals[0],
        raw_rmse_n=rmse_vals[1],
        raw_r2_Ea=r2_vals[2],
        r2_logA  =r2_logA,
        pct_err_A=pct_err_A
    )

# -----------------------------------------------------------------------------
# 4) Define search space & launch tuner
# -----------------------------------------------------------------------------
search_space = {
    "depth": tune.qrandint(2, 6, 1),
    "message_hidden_dim": tune.qrandint(256, 1024, 64),
    "ffn_hidden_dim": tune.qrandint(256, 2048, 128),
    "ffn_num_layers": tune.qrandint(1, 4, 1),
    "dropout": tune.uniform(0.0, 0.5),
    "batch_size": tune.choice([32, 64, 128]),
    "warmup_epochs": tune.choice([1, 2, 5]),
    "init_lr": tune.loguniform(1e-5, 1e-3),
    "max_lr":  tune.loguniform(1e-4, 1e-2),
    "final_lr":tune.loguniform(1e-6, 1e-4),
    "max_epochs": tune.choice([50, 100, 200])
}

ray.init(ignore_reinit_error=True)
scheduler     = FIFOScheduler()
search_alg    = HyperOptSearch(n_initial_points=5, random_state_seed=42)
scaling_cfg   = ScalingConfig(num_workers=1, use_gpu=False)
ckpt_cfg      = CheckpointConfig(num_to_keep=1,
                                 checkpoint_score_attribute="val_loss",
                                 checkpoint_score_order="min")
run_cfg       = RunConfig(checkpoint_config=ckpt_cfg,
                          storage_path="hpopt/ray_results")

torch_trainer = TorchTrainer(
    train_loop_per_worker=train_model,
    train_loop_config=search_space,
    scaling_config=scaling_cfg,
    run_config=run_cfg
)

tuner = Tuner(
    torch_trainer,
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=20
    )
)

results = tuner.fit()

# -----------------------------------------------------------------------------
# 5) Inspect all trials & the best one
# -----------------------------------------------------------------------------
df    = results.get_dataframe()
best  = results.get_best_result(metric="val_loss", mode="min")

print("=== All trials summary ===")
print(df[["trial_id", "val_loss", "raw_mae_A", "raw_rmse_n", "raw_r2_Ea", "r2_logA", "pct_err_A"]])

print("\n=== Best trial ===")
print("Trial ID:      ", best.trial_id)
print("Config:        ", best.config["train_loop_config"])
print("Best checkpoint path:", Path(best.checkpoint.path) / "checkpoint.ckpt")
