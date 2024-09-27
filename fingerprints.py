import numpy as np
from rdkit import Chem
import os
from rdkit.Chem.AtomPairs.Pairs import *


def make_atompair_fingerprints():
    fingerprints = []
    for file in os.listdir('/project2/andrewferguson/berlaga/drugdiscovery/ligands'):
        if '_smi.txt' in file and os.path.isfile(f'/project2/andrewferguson/berlaga/drugdiscovery/ligands/{file}'):
            smiles_string = open(f'/project2/andrewferguson/berlaga/drugdiscovery/ligands/{file}', 'r').readlines()[0]
            mol = Chem.MolFromSmiles(smiles_string)
            fp1 = GetHashedAtomPairFingerprint(mol)
            fingerprints.append(np.array([b for b in fp1]))
    fingerprints = np.array(fingerprints)
    # fingerprints = fingerprints[:, ~np.all(fingerprints == 0, axis=0)]
    # if len(np.unique(fingerprints, axis=0)) < len(fingerprints):
    #     print(len(np.unique(fingerprints, axis=0)))
    #     print("ERROR")
    np.save("ap_fingerprints.npy", fingerprints)