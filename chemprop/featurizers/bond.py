from typing import Sequence, List

import numpy as np
from rdkit.Chem.rdchem import Bond, BondType

from chemprop.featurizers.base import VectorFeaturizer


class MultiHotBondFeaturizer(VectorFeaturizer[Bond]):
    """A :class:`MultiHotBondFeaturizer` feauturizes bonds based on the following attributes:

    * ``null``-ity (i.e., is the bond ``None``?)
    * bond type
    * conjugated?
    * in ring?
    * stereochemistry

    The feature vectors produced by this featurizer have the following (general) signature:

    +---------------------+-----------------+--------------+
    | slice [start, stop) | subfeature      | unknown pad? |
    +=====================+=================+==============+
    | 0-1                 | null?           | N            |
    +---------------------+-----------------+--------------+
    | 1-5                 | bond type       | N            |
    +---------------------+-----------------+--------------+
    | 5-6                 | conjugated?     | N            |
    +---------------------+-----------------+--------------+
    | 6-8                 | in ring?        | N            |
    +---------------------+-----------------+--------------+
    | 7-14                | stereochemistry | Y            |
    +---------------------+-----------------+--------------+

    **NOTE**: the above signature only applies for the default arguments, as the bond type and
    sterochemistry slices can increase in size depending on the input arguments.

    Parameters
    ----------
    bond_types : Sequence[BondType] | None, default=[SINGLE, DOUBLE, TRIPLE, AROMATIC]
        the known bond types
    stereos : Sequence[int] | None, default=[0, 1, 2, 3, 4, 5]
        the known bond stereochemistries. See [1]_ for more details

    References
    ----------
    .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values
    """

    def __init__(
        self, bond_types: Sequence[BondType] | None = None, stereos: Sequence[int] | None = None
    ):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        self.stereo = stereos or range(6)

    def __len__(self):
        return 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2

        stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
        x[i + stereo_bit] = 1

        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1


class RIGRBondFeaturizer(VectorFeaturizer[Bond]):
    """A :class:`RIGRBondFeaturizer` feauturizes bonds based on only the resonance-invariant features:

    * ``null``-ity (i.e., is the bond ``None``?)
    * in ring?
    """

    def __len__(self):
        return 2

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        x[1] = int(b.IsInRing())

        return x

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)

        return xs.index(x) if x in xs else n, n + 1


class SHAPMultiHotBondFeaturizer(MultiHotBondFeaturizer):
    """A custom MultiHotBondFeaturizer that allows for selective feature ablation.

    Parameters
    ----------
    keep_features : List[bool], optional
        a list of booleans to indicate which bond features to keep except for nullity. If None, all features are kept. For any element that is False, the corresponding feature's encoding is set to all zeros. Useful for ablation and SHAP analysis.
    """

    def __init__(self,
                 bond_types: Sequence[BondType] | None = None,
                 stereos: Sequence[int] | None = None,
                 keep_features: List[bool] = None):
        super().__init__(bond_types, stereos)

        self._MultiHotBondFeaturizer__size = 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

        if keep_features is None:
            keep_features = [True] * 5
        self.keep_features = keep_features

    def __len__(self) -> int:
        return self._MultiHotBondFeaturizer__size

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x
        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if self.keep_features[0] and bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        if self.keep_features[1]:
            x[i] = int(b.GetIsConjugated())
        if self.keep_features[2]:
            x[i + 1] = int(b.IsInRing())
        i += 2

        if self.keep_features[3]:
            stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
            x[i + stereo_bit] = 1

        return x

    def zero_mask(self) -> np.ndarray:
        """Featurize the bond by setting all bits to zero."""
        return np.zeros(len(self), int)

    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        """Returns a tuple of the index of ``x`` in ``xs`` and ``len(xs) + 1`` if ``x`` is in ``xs``.
        Otherwise, returns a tuple with ``len(xs)`` and ``len(xs) + 1``."""
        n = len(xs)
        return xs.index(x) if x in xs else n, n + 1