import os
import sys
import ast

import pytest
from rdkit import Chem

habnet_path = "/home/calvin/code/HAbstractionNet"
chemprop_path = "/home/calvin/code/chemprop_phd_customised"
sys.path.append(chemprop_path)
sys.path.append(habnet_path)

from chemprop import utils, data
from featuriser import featurise

#TODO: Speak to Alon about the Smiles Generated

class TestFeaturise:

    def setup_class(self):
        self.sdf = "/home/calvin/code/HAbstractionNet/data/processed/sdf_files/rmg_rxn_2.sdf"
        self.keep_h = True
        self.add_h = False
        self.mol_type = "r1h"
        self.mol_type_other = "r2h"

    def test_make_mol_from_sdf(self):
        mol = utils.make_mol_from_sdf(self.sdf, self.keep_h, self.add_h, self.mol_type)
        assert mol is not None
        assert isinstance(mol, list) or isinstance(mol, Chem.Mol)
        assert mol.GetProp("type") == self.mol_type
        assert mol.GetProp("reaction") == "rmg_rxn_2"
        assert ast.literal_eval(mol.GetProp("mol_properties")) == {"0": {"label": "donator", "atom_type": "N3s"}, "4": {"label": "d_hydrogen", "atom_type": "H0"}}
        assert Chem.MolToSmiles(mol) == "[H][N]N([H])[H]"

        mol_2 = utils.make_mol_from_sdf(self.sdf, self.keep_h, self.add_h, self.mol_type_other)
        assert mol_2 is not None
        assert isinstance(mol_2, list) or isinstance(mol_2, Chem.Mol)
        assert mol_2.GetProp("type") == self.mol_type_other
        assert mol_2.GetProp("reaction") == "rmg_rxn_2"
        assert ast.literal_eval(mol_2.GetProp("mol_properties")) == {"2": {"label": "acceptor", "atom_type": "N3s"}, "8": {"label": "a_hydrogen", "atom_type": "H0"}}
        assert Chem.MolToSmiles(mol_2) == "[H][N]C([H])([H])C([H])([H])[H]"

    def test_get_sdf_files(self):
        sdf_files = featurise.get_sdf_files(path = "/home/calvin/code/HAbstractionNet/data/processed/sdf_files")

        assert isinstance(sdf_files, list)
        assert '/home/calvin/code/HAbstractionNet/data/processed/sdf_files/rmg_rxn_2.sdf' in sdf_files
        assert len(sdf_files) == 1485

    def test_load_datapoints(self):
    
        sdf_files = ['/home/calvin/code/HAbstractionNet/data/processed/sdf_files/rmg_rxn_86.sdf', '/home/calvin/code/HAbstractionNet/data/processed/sdf_files/rxn_191.sdf']
        datapoints = featurise.load_datapoints(sdf_files, [self.mol_type, self.mol_type_other], self.keep_h, self.add_h)

        assert isinstance(datapoints, list)
        assert len(datapoints) == 2
        assert len(datapoints[0]) == 2
        assert len(datapoints[1]) == 2
        assert isinstance(datapoints[0][0], data.MoleculeDatapoint)
        assert isinstance(datapoints[1][0], data.MoleculeDatapoint)
        assert datapoints[0][0].mol.GetProp("type") == 'r1h'
        assert datapoints[1][0].mol.GetProp("reaction") == 'rmg_rxn_86'
        assert datapoints[0][0].name == 'rmg_rxn_86_r1h'
        assert Chem.MolToSmiles(datapoints[0][0].mol) == "[H]C1([H])OC([H])([H])C([H])([H])C([H])([H])O1"
        assert Chem.MolToSmiles(datapoints[1][0].mol) == '[H]S[H]'
