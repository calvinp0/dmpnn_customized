from dataclasses import InitVar, dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
import torch

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.featurizers.atom import SHAPMultiHotAtomFeaturizer
from chemprop.featurizers.bond import SHAPMultiHotBondFeaturizer


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0
     # RBF parameters
    rbf_D_min: float = 0.0
    rbf_D_max: float = 5.0
    rbf_D_count: int = 10
    rbf_gamma: float = 10.0


    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__()
        # parent dimensions
        base_atom_dim = self.atom_fdim
        base_bond_dim = self.bond_fdim

        # extras
        angle_dim = 3               # sin_mean, cos_mean, mask
        rbf_dim   = self.rbf_D_count

        # override dims
        self.atom_fdim = base_atom_dim + extra_atom_fdim
        self.bond_fdim = base_bond_dim + extra_bond_fdim + rbf_dim + angle_dim

    def _rbf(self, d: float) -> np.ndarray:
        # compute RBF for distance d
        mu = torch.linspace(self.rbf_D_min, self.rbf_D_max, self.rbf_D_count)
        rbf = torch.exp(-self.rbf_gamma * (d - mu)**2)
        return rbf.numpy()

    @staticmethod
    def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        ba /= np.linalg.norm(ba) + 1e-7
        bc /= np.linalg.norm(bc) + 1e-7
        cos_a = np.dot(ba, bc)
        return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    @classmethod
    def compute_edge_angle_features(cls, mol: Chem.Mol) -> Dict[Tuple[int,int], np.ndarray]:
        conf = mol.GetConformer()
        feats: Dict[Tuple[int,int], np.ndarray] = {}
        for center in range(mol.GetNumAtoms()):
            nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()]
            if len(nbrs) < 2:
                continue
            for i in nbrs:
                angles = []
                for k in nbrs:
                    if k == i:
                        continue
                    pos_i = np.array(conf.GetAtomPosition(i))
                    pos_c = np.array(conf.GetAtomPosition(center))
                    pos_k = np.array(conf.GetAtomPosition(k))
                    angles.append(cls.compute_angle(pos_i, pos_c, pos_k))
                if angles:
                    feats[(i, center)] = np.array([np.mean(np.sin(angles)), np.mean(np.cos(angles)), 1.0])
                else:
                    feats[(i, center)] = np.array([0.0, 0.0, -10.0])
        return feats

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        # validate extras
        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(f"Expected {n_atoms} atom extras, got {len(atom_features_extra)}")
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(f"Expected {n_bonds} bond extras, got {len(bond_features_extra)}")

        # node features
        V = np.zeros((1, self.atom_fdim), dtype=np.float32) if n_atoms == 0 else \
            np.stack([self.atom_featurizer(a) for a in mol.GetAtoms()])
        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        # edge containers
        E = np.zeros((2 * n_bonds, self.bond_fdim), dtype=np.float32)
        edge_index = [[], []]
        angle_feats = self.compute_edge_angle_features(mol)
        conf = mol.GetConformer()

        i = 0
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]))

            # distance + RBF
            pos_u = np.array(conf.GetAtomPosition(u))
            pos_v = np.array(conf.GetAtomPosition(v))
            dist = np.linalg.norm(pos_u - pos_v)
            rbf_feat = self._rbf(dist)

            # angle features
            ang_uv = angle_feats.get((u, v), np.array([0.0, 0.0, -10.0]))
            ang_vu = angle_feats.get((v, u), np.array([0.0, 0.0, -10.0]))

            # full feature vectors
            feat_uv = np.concatenate((x_e, rbf_feat, ang_uv), axis=0)
            feat_vu = np.concatenate((x_e, rbf_feat, ang_vu), axis=0)

            # sanity check
            assert feat_uv.shape[0] == self.bond_fdim, \
                f"feat length {feat_uv.shape[0]} != bond_fdim {self.bond_fdim}"

            # assign per direction
            E[i]   = feat_uv
            E[i+1] = feat_vu

            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])
            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, dtype=int)

        return MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)

@dataclass
class SHAPSimpleMoleculeMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """A custom SimpleMoleculeMolGraphFeaturizer with additional feature control."""

    keep_atom_features: Optional[List[bool]] = None
    keep_bond_features: Optional[List[bool]] = None
    keep_atoms: Optional[List[bool]] = None
    keep_bonds: Optional[List[bool]] = None

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__(extra_atom_fdim, extra_bond_fdim)

        if isinstance(self.atom_featurizer, SHAPMultiHotAtomFeaturizer) and self.keep_atom_features is not None:
            self.atom_featurizer.keep_features = self.keep_atom_features
        if isinstance(self.bond_featurizer, SHAPMultiHotBondFeaturizer) and self.keep_bond_features is not None:
            self.bond_featurizer.keep_features = self.keep_bond_features

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if self.keep_atoms is None:
            self.keep_atoms = [True] * n_atoms
        if self.keep_bonds is None:
            self.keep_bonds = [True] * n_bonds

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )
        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) if self.keep_atoms[a.GetIdx()] else self.atom_featurizer.zero_mask()
                          for a in mol.GetAtoms()], dtype=np.single)

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                x_e = self.bond_featurizer(bond) if self.keep_bonds[bond.GetIdx()] else self.bond_featurizer.zero_mask()

                if bond_features_extra is not None:
                    x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

                E[i: i + 2] = x_e
                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])
                i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)
        return MolGraph(V, E, edge_index, rev_edge_index)
