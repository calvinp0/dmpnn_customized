# ──────────────────────────────────────────────────────────────────────────────
# dmpnn_optuna.py     •  clean, modular variant with cached preprocessing
# ──────────────────────────────────────────────────────────────────────────────
import argparse, json, random, logging
from pathlib import Path
from functools import partial
import os
import numpy as np
import pandas as pd
import torch
import optuna
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score)

from chemprop import data, featurizers, nn
from chemprop.models import multi
from chemprop.nn import metrics
from chemprop.CUSTOM.featuriser.featurise import Featuriser, MOL_TYPES         # ⇦ your custom code
from chemprop.nn.metrics import ChempropMetric, LossFunctionRegistry

# ─── globals ──────────────────────────────────────────────────────────────────
ROOT        = Path("~/Code/dmpnn_customized").expanduser()
RAW_CSV     = ROOT / "DATA/target_data/kinetics_summary.csv"
PROC_CSV    = ROOT / "DATA/target_data/temp_target_kinetic_data.csv"
ATOM_FEATS  = ROOT / "DATA/sdf_data/all_sdf_features.csv"
SEED        = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

# ── fix: drop unused “featurizer” arg in build_dataloaders  ──────────────────
def build_dataloaders(train, val, test, bs):
    gen = torch.Generator().manual_seed(SEED)
    train_loader = data.build_dataloader(
        train, batch_size=bs, shuffle=True,
        generator=gen, num_workers=8,
        persistent_workers=True, pin_memory=True
    )
    common = dict(shuffle=False, batch_size=min(64, bs), num_workers=4)
    val_loader  = data.build_dataloader(val,  **common)
    test_loader = data.build_dataloader(test, **common)
    return train_loader, val_loader, test_loader


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


def attach_morgan_to_dps(dps,atom_extra_feats ):
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

def get_atom_feats_for_dp(dp_name, atom_extra_feats):
    rxn_id, mol_type = dp_name.rsplit('_', 1)
    subset = atom_extra_feats[
        (atom_extra_feats['rxn_id'] == rxn_id) &
        (atom_extra_feats['mol_type'] == mol_type)
    ]
    # Sort by focus_atom_idx
    subset = subset.sort_values("focus_atom_idx")
    return subset


def create_model(featurizer, trial):
    depth              = trial.suggest_int('depth', 2, 5)
    msg_dim            = trial.suggest_int('msg_dim', 256, 1024, log=True)
    ffn_dim            = trial.suggest_int('ffn_dim', 256, 1024, log=True)
    ffn_layers         = trial.suggest_int('ffn_layers', 1, 3)
    dropout            = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
    shared_encoder     = trial.suggest_categorical('shared', [True, False])
    task_weights       = trial.suggest_categorical('task_w',
                             [[10,1,1], [5,1,1], [20,1,1]])
    delta              = trial.suggest_float('delta', 0.2, 1.0, step=0.2)
    use_bn  = trial.suggest_categorical('batch_norm', [True, False])
    agg_name = trial.suggest_categorical('aggregation',['mean', 'sum', 'norm', 'attn'])

    blocks = [nn.BondMessagePassing(depth=depth, dropout=dropout,
                                    d_v=featurizer.atom_fdim,
                                    d_e=featurizer.bond_fdim,
                                    d_h=msg_dim)
              for _ in MOL_TYPES]
    mcmp = nn.MulticomponentMessagePassing(blocks=blocks,
                                           n_components=len(MOL_TYPES),
                                           shared=shared_encoder)
    # -------- aggregation ----------------------------------------------------
    component_dim = mcmp.blocks[0].output_dim
    if agg_name == 'mean':
        agg = nn.MeanAggregation()
    elif agg_name == 'sum':
        agg = nn.SumAggregation()
    elif agg_name == 'norm':
        agg = nn.NormAggregation(norm=trial.suggest_float('norm_c', 10.0, 200.0, log=True))
    else:  # 'attn'
        agg = nn.AttentiveAggregation(output_size=component_dim)

    model = multi.MulticomponentMPNN(
        message_passing=mcmp,
        agg=agg,
        predictor=nn.RegressionFFN(
            n_tasks=3,
            input_dim=mcmp.output_dim,
            hidden_dim=ffn_dim,
            n_layers=ffn_layers,
            dropout=dropout,
            criterion=HuberMSE(task_weights=task_weights, delta=delta)
        ),
        metrics=[metrics.RMSE(task_weights=task_weights),
                 metrics.MAE(task_weights=task_weights),
                 metrics.R2Score(task_weights=task_weights)],
        warmup_epochs=trial.suggest_int('warmup', 0, 5),
        init_lr=trial.suggest_float('init_lr', 1e-4, 1e-3, log=True),
        max_lr =trial.suggest_float('max_lr',  1e-3, 1e-2, log=True),
        final_lr=trial.suggest_float('final_lr',1e-5,1e-4, log=True),
        batch_norm=use_bn
    )
    return model



# ─── preprocessing (only once!) ───────────────────────────────────────────────
def preprocess() -> tuple[data.MulticomponentDataset,
                          data.MulticomponentDataset,
                          data.MulticomponentDataset,
                          nn.UnscaleTransform,
                          PowerTransformer]:
    df = pd.read_csv(RAW_CSV)
    df = (df.query("label == 'k_for (TST+T)'")
             .dropna(subset=['rxn', 'A', 'n', 'Ea'])
             .drop_duplicates(subset=['rxn'])
             .query("A > 0"))
    df['A_log10'] = np.log10(df['A'])
    pt_ea = PowerTransformer(method='yeo-johnson')
    df['Ea_yj'] = pt_ea.fit_transform(df['Ea'].to_numpy()[:,None]).ravel()
    df.to_csv(PROC_CSV, index=False)

    feat_data = Featuriser(
    sdf_path       = ROOT / "DATA/sdf_data",
    path          = PROC_CSV,                     # << correct keyword
    filter_rules  = {'label': 'k_for (TST+T)'},
    target_col    = ['A_log10', 'n', 'Ea_yj'],
    set_col_index = False,                        # << keep the ‘rxn’ column!
    include_extra_features = False
)

    mols      = [d.mol for d in feat_data[0]]
    splits    = data.make_split_indices(mols, "kennard_stone", (0.8,0.1,0.1))
    ds_train, ds_val, ds_test = data.split_data_by_indices(feat_data, *splits)

    # extra atom features
    rxn_ids = []
    for i in range(len(ds_train[0][0])):
        rxn_name = ds_train[0][0][i].name
        # Need to remove _r1h or _r2h from the reaction name
        rxn_name = rxn_name.replace('_r1h', '').replace('_r2h', '')
        rxn_ids.append(rxn_name)
    atom_extra_feats = pd.read_csv(ATOM_FEATS)
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
        
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=26)
# ---------- helper ----------------------------------------------------------
    def wrap(component_slice):
        """component_slice is the list of MoleculeDatapoint for one mol-type."""
        return data.MoleculeDataset(
            attach_morgan_to_dps(component_slice, atom_extra_feats),   # <- call once
            featurizer
        )

    # ---------- build per-component datasets ------------------------------------
    train_sets = [wrap(ds_train[0][i]) for i in range(len(MOL_TYPES))]
    val_sets   = [wrap(ds_val[0][i])   for i in range(len(MOL_TYPES))]
    test_sets  = [wrap(ds_test[0][i])  for i in range(len(MOL_TYPES))]

    train_mcd  = data.MulticomponentDataset(train_sets)
    scaler     = train_mcd.normalize_targets()
    val_mcd    = data.MulticomponentDataset(val_sets);  val_mcd.normalize_targets(scaler)
    test_mcd   = data.MulticomponentDataset(test_sets); test_mcd.normalize_targets(scaler)

    return train_mcd, val_mcd, test_mcd, nn.UnscaleTransform.from_standard_scaler(scaler), pt_ea


# ─── optuna objective (pure training loop) ────────────────────────────────────

def objective(train_mcd, val_mcd, test_mcd, unscale, pt_ea, wandb_name, trial)
    seed_everything(SEED + trial.number)
    os.environ['WANDB_MODE'] = 'offline'

    # 1) INIT & CONFIG
    run = wandb.init(project=wandb_name, mode="offline",
               name=f"trial_{trial.number}", config=trial.params, reinit=True)
    wandb.config.update(trial.params, allow_val_change=True)

    # 2) BUILD DATA
    bs = trial.suggest_int('batch_size', 32, 128, step=32)
    tl, vl, xl = build_dataloaders(train_mcd, val_mcd, test_mcd, bs)

    # 3) MODEL & LOGGER
    model = create_model(train_mcd.datasets[0].featurizer, trial)
    wandb_logger = WandbLogger(
        project='dmpnn_customized',
        name=f"trial_{trial.number}",
        mode='offline'
    )
    wandb_logger.watch(model, log='all', log_freq=100)

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor='val/rmse', mode='min', save_top_k=1
    )
    early_stop = pl.callbacks.EarlyStopping(
        monitor='val/rmse', patience=8, mode='min'
    )
    trainer = pl.Trainer(
        accelerator="auto", devices=1,
        max_epochs=trial.suggest_int('epochs', 20, 120, step=10),
        logger=wandb_logger,
        enable_progress_bar=False,
        callbacks=[ckpt, early_stop]
    )

    # 4) TRAIN & TEST
    trainer.fit(model, tl, vl)
    test_metrics = trainer.test(model, xl, verbose=False)[0]
    test_rmse, test_mae, test_r2 = (
        test_metrics['test/rmse'],
        test_metrics['test/mae'],
        test_metrics['test/r2']
    )

    # 5) POST-PROCESS PREDICTIONS
    model.eval()
    with torch.no_grad():
        scaled_preds = torch.cat(trainer.predict(model, dataloaders=xl), dim=0).cpu().numpy()
    scaler = unscale.to_standard_scaler()
    unscaled_pred = scaler.inverse_transform(scaled_preds)
    y_scaled = torch.cat([b.Y for b in xl], dim=0).cpu().numpy()
    unscaled_true = scaler.inverse_transform(y_scaled)

    # revert log10 and YJ
    logA_pred, n_pred, Ea_yj_pred = unscaled_pred.T
    logA_true, n_true, Ea_yj_true = unscaled_true.T
    A_pred  = np.clip(10**logA_pred, 1e-8, None)
    A_true  = 10**logA_true
    Ea_pred = np.clip(pt_ea.inverse_transform(Ea_yj_pred[:,None]).ravel(), 0.0, None)
    Ea_true = pt_ea.inverse_transform(Ea_yj_true[:,None]).ravel()

    preds_final = np.vstack([A_pred, n_pred, Ea_pred]).T
    trues_final = np.vstack([A_true, n_true, Ea_true]).T

    # 6) LOG CUSTOM METRICS
    wandb.log({
        'test_rmse': test_rmse,
        'test_mae':  test_mae,
        'test_r2':   test_r2,
        'mae_A':     mean_absolute_error(trues_final, preds_final, multioutput='raw_values')[0],
        'rmse_A':    root_mean_squared_error(trues_final, preds_final, multioutput='raw_values')[0],
        'r2_A':      r2_score(trues_final, preds_final, multioutput='raw_values')[0],
    })

    # 7) ERROR-ANALYSIS TABLE
    table = wandb.Table(columns=[
        'A_true','A_pred','err_A',
        'n_true','n_pred','err_n',
        'Ea_true','Ea_pred','err_Ea'
    ])
    for t,p in zip(trues_final, preds_final):
        table.add_data(
            t[0], p[0], abs(t[0]-p[0]),
            t[1], p[1], abs(t[1]-p[1]),
            t[2], p[2], abs(t[2]-p[2])
        )
    wandb.log({'error_table': table})

    # 8) HISTOGRAMS
    wandb.log({
        'error_histogram': wandb.Histogram((preds_final-trues_final).flatten()),
        'A_pred_dist':     wandb.Histogram(A_pred),
    })

    # 9) ARTIFACTS
    model_art = wandb.Artifact('dmpnn-model', type='model')
    model_art.add_file(ckpt.best_model_path)
    wandb.log_artifact(model_art)

    data_art = wandb.Artifact('kinetics-processed', type='dataset')
    data_art.add_file(PROC_CSV)
    wandb.log_artifact(data_art)

    run.finish()
    return test_rmse

# ─── entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--wandb_name", type=str, default="dmpnn_customized")
    parser.add_argument("--optuna_db", type=str, default="study.db")
    args = parser.parse_args()

    seed_everything()

    train_mcd, val_mcd, test_mcd, unscale, pt_ea = preprocess()
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(5),
        storage=f"sqlite:///{args.optuna_db}",
        load_if_exists=True
    )
    func = partial(objective, train_mcd, val_mcd, test_mcd, unscale, pt_ea, args.wandb_name)
    study.optimize(func, n_trials=args.trials)

    best = study.best_trial
    json.dump({
        "number": best.number,
        "value": best.value,
        "params": best.params,
        "ckpt": best.user_attrs["ckpt"]
    }, open("best_trial.json", "w"), indent=2)

    print(f"🏆 Trial {best.number}  RMSE={best.value:.4f}")

if __name__ == "__main__":
    main()
