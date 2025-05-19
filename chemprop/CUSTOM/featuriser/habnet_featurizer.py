import numpy as np
from chemprop.featurizers.atom import MultiHotAtomFeaturizer

class AtomHAbNetFeaturizer:
    def __init__(self, zmat_data: bool = False):
        self.base_featurizer = MultiHotAtomFeaturizer.v2()
        # Extra features: [d_hydrogen, a_hydrogen, is_donor, is_acceptor, electro_R, electro_A, electro_D]
        self.extra_fdim = 10 if zmat_data else 4
        self.electro_map = {}
        self.mol_properties = {}
        self.zmat_data = zmat_data

    def set_mol(self, mol):
        """
        Load molecule-level properties from the RDKIT molecule object.
        """
        if mol.HasProp('mol_properties'):
            self.mol_properties = mol.GetProp('mol_properties')
        else:
            self.mol_properties = {}

    def __len__(self):
        # Total feature dimension = base v2 dimension + 4 extra features
        return len(self.base_featurizer) + self.extra_fdim

    def __call__(self, atom):
        # Get the base v2 features
        base_feats = self.base_featurizer(atom)
        # Initialize extra features vector (all zeros by default)
        extra_feats = np.zeros(self.extra_fdim, dtype=np.float32)

        # Use molecule-level properties to set donor/acceptor features.
        # mol_properties is expected to be a dict with atom indices (as strings) as keys.
        # Check for atom-level properties attached during molecule creation.
        if atom.HasProp("d_hydrogen") and atom.GetProp("d_hydrogen") in ["True", "1"]:
            extra_feats[0] = 1.0
        if atom.HasProp("a_hydrogen") and atom.GetProp("a_hydrogen") in ["True", "1"]:
            extra_feats[1] = 1.0
        if atom.HasProp("is_donor") and atom.GetProp("is_donor") in ["True", "1"]:
            extra_feats[2] = 1.0
        if atom.HasProp("is_acceptor") and atom.GetProp("is_acceptor") in ["True", "1"]:
            extra_feats[3] = 1.0

        # If zmat_data is enabled, add electro mapping values and presence indicators.
        if self.zmat_data:
            # Helper to process a property: returns (value, present_flag)
            def get_electro_props(prop_name):
                if atom.HasProp(prop_name):
                    val_str = atom.GetProp(prop_name)
                    if val_str and val_str.lower() != 'none':
                        try:
                            return float(val_str), 1.0
                        except ValueError:
                            # Log or handle the conversion error as needed.
                            return 0.0, 0.0
                return 0.0, 0.0

            # Process electro_R, electro_A, and electro_D.
            r_val, r_flag = get_electro_props('electro_R')
            a_val, a_flag = get_electro_props('electro_A')
            d_val, d_flag = get_electro_props('electro_D')

            # Place them in the feature vector.
            extra_feats[4] = r_val
            extra_feats[5] = r_flag
            extra_feats[6] = a_val
            extra_feats[7] = a_flag
            extra_feats[8] = d_val
            extra_feats[9] = d_flag
        
        # Concatenate base and extra features
        return np.concatenate([base_feats, extra_feats])
