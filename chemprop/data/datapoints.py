from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from rdkit.Chem import AllChem as Chem

from chemprop.featurizers import Featurizer
from chemprop.utils import make_mol, make_mol_from_sdf

MoleculeFeaturizer = Featurizer[Chem.Mol, np.ndarray]


@dataclass(slots=True)
class _DatapointMixin:
    """A mixin class for both molecule- and reaction- and multicomponent-type data"""

    y: np.ndarray | None = None
    """the targets for the molecule with unknown targets indicated by `nan`s"""
    weight: float = 1.0
    """the weight of this datapoint for the loss calculation."""
    gt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form `<x`"""
    lt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form `>x`"""
    x_d: np.ndarray | None = None
    """A vector of length ``d_f`` containing additional features (e.g., Morgan fingerprint) that
    will be concatenated to the global representation *after* aggregation"""
    x_phase: list[float] = None
    """A one-hot vector indicating the phase of the data, as used in spectra data."""
    name: str | None = None
    """A string identifier for the datapoint."""

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.x_d is not None:
            self.x_d[np.isnan(self.x_d)] = NAN_TOKEN

    @property
    def t(self) -> int | None:
        return len(self.y) if self.y is not None else None


@dataclass
class _MoleculeDatapointMixin:
    mol: Chem.Mol
    """the molecule associated with this datapoint"""

    @classmethod
    def from_smi(
        cls, smi: str, *args, keep_h: bool = False, add_h: bool = False,  **kwargs
    ) -> _MoleculeDatapointMixin:
        mol = make_mol(smi, keep_h, add_h)

        kwargs["name"] = smi if "name" not in kwargs else kwargs["name"]

        return cls(mol, *args, **kwargs)

    @classmethod
    def from_sdf(
        cls, sdf: str, *args, keep_h: bool = False, add_h: bool = False, mol_type: str = "all", include_extra_features: bool = False, **kwargs
    ) -> _MoleculeDatapointMixin:
        mol = make_mol_from_sdf(sdf, keep_h, add_h, mol_type)

        mol = attach_molecule_properties(mol)

        reaction = mol.GetProp("reaction")
        kwargs["name"] = str(reaction) + "_" + str(mol_type) if "name" not in kwargs else kwargs["name"]
        if include_extra_features:
            kwargs["V_f"] = build_extra_features(mol)

        return cls(mol, *args, **kwargs)


def build_extra_features(mol):
    """
    Build an extra feature matrix for the molecule using its mol_properties and electro_map.
    Assumes mol_properties and electro_map are stored as JSON strings on the molecule.
    
    Returns:
        np.ndarray of shape (n_atoms, 10)
    """
    n_atoms = mol.GetNumAtoms()
    extra_dim = 10  # 4 from mol_properties, 6 from electro_map (value and flag for each)
    features = np.zeros((n_atoms, extra_dim), dtype=np.float32)
    
    # Parse molecule-level properties from the JSON strings (if present)
    try:
        mol_props = json.loads(mol.GetProp('mol_properties')) if mol.HasProp('mol_properties') else {}
    except Exception:
        mol_props = {}
    try:
        elec_map = json.loads(mol.GetProp('electro_map')) if mol.HasProp('electro_map') else {}
    except Exception:
        elec_map = {}
    
    for i in range(n_atoms):
        idx = str(i)
        # Process mol_properties: assign donor/acceptor flags to slots 0–3.
        if idx in mol_props:
            label = mol_props[idx].get("label", "").lower()
            if label == "d_hydrogen":
                features[i, 0] = 1.0
            elif label == "a_hydrogen":
                features[i, 1] = 1.0
            elif label == "donator":
                features[i, 2] = 1.0
            elif label == "acceptor":
                features[i, 3] = 1.0
        
        # Define a helper to process each electro property.
        def get_electro_prop(prop_name):
            if idx in elec_map:
                val = elec_map[idx].get(prop_name)
                if val not in [None, "None"]:
                    try:
                        return float(val), 1.0
                    except ValueError:
                        # Could log a warning here if needed.
                        return 0.0, 0.0
            return 0.0, 0.0

        # Process the electro_map properties.
        r_val, r_flag = get_electro_prop("R")
        a_val, a_flag = get_electro_prop("A")
        d_val, d_flag = get_electro_prop("D")
        
        # Place them in the feature vector.
        features[i, 4] = r_val
        features[i, 5] = r_flag
        features[i, 6] = a_val
        features[i, 7] = a_flag
        features[i, 8] = d_val
        features[i, 9] = d_flag

    return features



def attach_molecule_properties(mol):
    """
    Load molecule-level properties from the RDKIT molecule object.

    Parameters
    ----------
    mol : Chem.Mol
        an RDKit molecule object.

    Returns
    -------
    dict
        a dictionary containing the molecule-level properties.
    """
    if mol.HasProp("mol_properties"):
        try:
            mol_properties = json.loads(mol.GetProp("mol_properties"))
        except Exception as e:
            raise ValueError(f"Error loading mol_properties: {e}")
        for atom in mol.GetAtoms():
            idx = str(atom.GetIdx())
            if idx in mol_properties:
                label = mol_properties[idx].get("label", "")
                # Attach properties as atom-level flags
                if label == "d_hydrogen":
                    atom.SetProp("d_hydrogen", "True")
                elif label == "a_hydrogen":
                    atom.SetProp("a_hydrogen", "True")
                elif label == "donator":
                    atom.SetProp("is_donor", "True")
                elif label == "acceptor":
                    atom.SetProp("is_acceptor", "True")
    if mol.HasProp("electro_map"):
        try:
            electro_map = json.loads(mol.GetProp("electro_map"))
        except Exception as e:
            raise ValueError(f"Error loading electro_map: {e}")
        for atom in mol.GetAtoms():
            idx = str(atom.GetIdx())
            if idx in electro_map:
                atom.SetProp("electro_R", str(electro_map[idx]["R"]))
                atom.SetProp("electro_A", str(electro_map[idx]["A"]))
                atom.SetProp("electro_D", str(electro_map[idx]["D"]))
    return mol


@dataclass
class MoleculeDatapoint(_DatapointMixin, _MoleculeDatapointMixin):
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    V_f: np.ndarray | None = None
    """a numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
    ``d_vf`` is the number of additional features that will be concatenated to atom-level features
    *before* message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of bonds in the molecule, and
    ``d_ef`` is the number of additional features  containing additional features that will be
    concatenated to bond-level features *before* message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of atoms in the molecule, and
    ``d_vd`` is the number of additional descriptors that will be concatenated to atom-level
    descriptors *after* message passing"""

    def __post_init__(self):
        NAN_TOKEN = 0
        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__()

    def __len__(self) -> int:
        return 1


@dataclass
class _ReactionDatapointMixin:
    rct: Chem.Mol
    """the reactant associated with this datapoint"""
    pdt: Chem.Mol
    """the product associated with this datapoint"""

    @classmethod
    def from_smi(
        cls,
        rxn_or_smis: str | tuple[str, str],
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        **kwargs,
    ) -> _ReactionDatapointMixin:
        match rxn_or_smis:
            case str():
                rct_smi, agt_smi, pdt_smi = rxn_or_smis.split(">")
                rct_smi = f"{rct_smi}.{agt_smi}" if agt_smi else rct_smi
                name = rxn_or_smis
            case tuple():
                rct_smi, pdt_smi = rxn_or_smis
                name = ">>".join(rxn_or_smis)
            case _:
                raise TypeError(
                    "Must provide either a reaction SMARTS string or a tuple of reactant and"
                    " a product SMILES strings!"
                )

        rct = make_mol(rct_smi, keep_h, add_h)
        pdt = make_mol(pdt_smi, keep_h, add_h)

        kwargs["name"] = name if "name" not in kwargs else kwargs["name"]

        return cls(rct, pdt, *args, **kwargs)


@dataclass
class ReactionDatapoint(_DatapointMixin, _ReactionDatapointMixin):
    """A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets."""

    def __post_init__(self):
        if self.rct is None:
            raise ValueError("Reactant cannot be `None`!")
        if self.pdt is None:
            raise ValueError("Product cannot be `None`!")

        return super().__post_init__()

    def __len__(self) -> int:
        return 2
