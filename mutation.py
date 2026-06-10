from rdkit import Chem
from rdkit.Chem import RWMol
import random

def add_carbon(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    rw = RWMol(mol)

    atom_idx = random.randint(
        0,
        rw.GetNumAtoms()-1
    )

    new_atom = rw.AddAtom(
        Chem.Atom("C")
    )

    rw.AddBond(
        atom_idx,
        new_atom,
        Chem.BondType.SINGLE
    )

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)

    except:
        return None

def oxygen_to_nitrogen(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    rw = RWMol(mol)

    oxygens = [
        atom.GetIdx()
        for atom in rw.GetAtoms()
        if atom.GetSymbol()=="O"
    ]

    if len(oxygens)==0:
        return None

    idx = random.choice(oxygens)

    rw.GetAtomWithIdx(idx).SetAtomicNum(7)

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)

    except:
        return None

def add_fluorine(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    rw = RWMol(mol)

    atom_idx = random.randint(
        0,
        rw.GetNumAtoms()-1
    )

    f_idx = rw.AddAtom(
        Chem.Atom("F")
    )

    rw.AddBond(
        atom_idx,
        f_idx,
        Chem.BondType.SINGLE
    )

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)

    except:
        return None

mutation_functions = [
    add_carbon,
    oxygen_to_nitrogen,
    add_fluorine
]

candidates = []

for smi in seed_smiles:

    for _ in range(10):

        fn = random.choice(
            mutation_functions
        )

        try:

            new_smi = fn(smi)

            if new_smi:
                candidates.append(
                    new_smi
                )

        except:
            pass