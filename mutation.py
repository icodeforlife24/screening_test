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



import random
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import Descriptors

def electrolyte_filter(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)

    if mw < 50:
        return False

    if mol.GetNumAtoms() < 5:
        return False

    return True
def generate_candidates(
    seed_smiles,
    mutations_per_molecule=10
):

    mutation_functions = [
        add_carbon,
        oxygen_to_nitrogen,
        add_fluorine
    ]

    candidates = []

    for smi in seed_smiles:

        for _ in range(
            mutations_per_molecule
        ):

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

    candidates = list(set(candidates))

    valid = []

    for smi in candidates:

        mol = Chem.MolFromSmiles(smi)

        if mol:
            valid.append(smi)
    valid = [
    s for s in valid
    if electrolyte_filter(s)]
    return valid

