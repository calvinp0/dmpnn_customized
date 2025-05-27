from typing import Iterable

import torch
from torch import Tensor

from chemprop.data import BatchMolGraph, MulticomponentTrainingBatch
from chemprop.models.model import MPNN
from chemprop.nn import Aggregation, MulticomponentMessagePassing, Predictor
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.transforms import ScaleTransform


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
        order_aware: bool = False,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
        )
        self.message_passing: MulticomponentMessagePassing
        self.order_aware = order_aware
    def fingerprint(
        self,
        bmgs: Iterable[BatchMolGraph],
        V_ds: Iterable[Tensor | None],
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        if self.order_aware:
            H = torch.cat(Hs, dim=-1)  # Preserve molecule order
        else:
            # Symmetrize: average of concat in both orders
            H1 = torch.cat(Hs, dim=-1)
            H2 = torch.cat(Hs[::-1], dim=-1)
            H = 0.5 * (H1 + H2)

        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    def on_validation_model_eval(self) -> None:
        self.eval()
        for block in self.message_passing.blocks:
            block.V_d_transform.train()
            block.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def get_batch_size(self, batch: MulticomponentTrainingBatch) -> int:
        return len(batch[0][0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location, weights_only=False)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        if hparams["predictor"]["criterion"] is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (
                not torch.cuda.is_available() and metric.device.type != "cpu"
            ):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]
        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        return submodules, state_dict, hparams


import torch
import torch.nn.functional as F
from torch import nn, optim
from lightning import pytorch as pl
from chemprop.data import MulticomponentTrainingBatch
from chemprop.models.model import MPNN
from chemprop.nn.message_passing import MulticomponentMessagePassing
from chemprop.nn import Aggregation
from chemprop.nn.transforms import ScaleTransform


class MultiHeadMulticomponentMPNN(MulticomponentMPNN):
    def __init__(self,
                 message_passing: MulticomponentMessagePassing,
                 agg: Aggregation,
                 predictor: Predictor = None,
                 hidden_dim: int = 512,
                 dropout: float = 0.2,
                 wA: float = 5.0,
                 wn: float = 1.0,
                 wEa: float = 1.0,
                 huber_delta: float = 1.0,
                 **mpnn_kwargs):
        # <-- Note: use the multi-component base
        super().__init__(
            message_passing = message_passing,
            agg             = agg,
            predictor       = predictor,
            **mpnn_kwargs
        )

        D = self.message_passing.output_dim  # fingerprint dim

        # three separate heads
        self.head_A  = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.head_n  = nn.Sequential(
            nn.Linear(D, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.head_Ea = nn.Sequential(
            nn.Linear(D, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

        # loss weights & Huber δ
        self.wA, self.wn, self.wEa = wA, wn, wEa
        self.delta = huber_delta

    def forward(self, bmgs, V_ds, X_d=None):
        # calls the multi-component fingerprint under the hood
        H = self.fingerprint(bmgs, V_ds, X_d)  # (batch_size, D)
        yA  = self.head_A(H).squeeze(-1)
        yn  = self.head_n(H).squeeze(-1)
        yEa = self.head_Ea(H).squeeze(-1)
        return torch.stack([yA, yn, yEa], dim=1)

    def training_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, *rest = batch
        mask    = targets.isfinite()
        targets = targets.nan_to_num(0.0)

        preds = self(bmg, V_d, X_d)    # (B,3)
        A_p,   n_p,   Ea_p   = preds[:,0], preds[:,1], preds[:,2]
        A_t,   n_t,   Ea_t   = targets[:,0], targets[:,1], targets[:,2]

        # Huber on log10(A):
        diff = A_p - A_t
        absd = diff.abs()
        hub  = torch.where(
            absd <= self.delta,
            0.5 * diff**2,
            self.delta * (absd - 0.5*self.delta)
        ).mean()

        # MSE on n & Ea
        loss_n  = F.mse_loss(n_p,  n_t)
        loss_Ea = F.mse_loss(Ea_p, Ea_t)

        loss = (self.wA * hub + self.wn * loss_n + self.wEa * loss_Ea) / (self.wA + self.wn + self.wEa)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, *_ = batch

        # forward & split
        preds = self(bmg, V_d, X_d)            # (B,3)
        A_p,   n_p,   Ea_p   = preds.unbind(1)
        A_t,   n_t,   Ea_t   = targets.unbind(1)

        # compute the same loss you do in training_step
        diff = A_p - A_t
        hub  = torch.where(diff.abs() <= self.delta,
                        0.5 * diff**2,
                        self.delta * (diff.abs() - 0.5*self.delta)
                        ).mean()
        loss_n  = F.mse_loss(n_p,  n_t)
        loss_Ea = F.mse_loss(Ea_p, Ea_t)
        val_loss = (self.wA*hub + self.wn*loss_n + self.wEa*loss_Ea) / (self.wA + self.wn + self.wEa)

        # log it once‐per‐epoch
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)