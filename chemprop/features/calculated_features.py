from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
#from rmgpy.molecule import Molecule
import numpy as np


def morgan_fingerprint(mol: Chem.rdchem.Mol, radius=2, nBits=2048):
    """calculate morgan fingerprint, default radius 2"""
    vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(vec, features)
    # if inp.cuda:
    #     features = features.cuda()
    return features


def get_h_bond_acceptor_count(mol: Chem.rdchem.Mol):
    """CalcNumHBA includes N,O,S (counts 1 for O although 2 lone pairs)
    CalcNumLipinskiHBA includes O,N (counts 1 for O although 2 lone pairs)
    should also include F, so made my own procedure
    /10 for scaling"""
    return (AllChem.CalcNumLipinskiHBA(mol) + get_num_atoms("F", mol)) / 10.0


#def get_rmg_symmetry(mol: Chem.rdchem.Mol):
#    smiles = Chem.MolToSmiles(mol)
#    try:
#        rmg_mol = Molecule(smiles=smiles)
#        symmetry = np.log2(rmg_mol.calculate_symmetry_number())
#    except:
#        symmetry = 0.0
#    return symmetry


def get_h_bond_donor_count(mol: Chem.rdchem.Mol):
    """CalcNumHBD includes N,O,S (counts 1 for C-NH2)
    CalcNumLipinskiHBD includes O,N (counts 2 for C-NH2)
     /10 for scaling"""
    return AllChem.CalcNumLipinskiHBD(mol) / 10.0


def get_num_atoms(symbol, mol: Chem.rdchem.Mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            count += 1
    return count


def get_num_rotors(mol: Chem.rdchem.Mol):
    """/10 for scaling"""
    return AllChem.CalcNumRotatableBonds(mol) / 10.0


def topological_polar_surface_area(mol: Chem.rdchem.Mol):
    """/100 for scaling"""
    return AllChem.CalcTPSA(mol) / 100.0


def molecular_radius(mol: Chem.rdchem.Mol):
    """/100 for scaling"""
    return Crippen.MolMR(mol) / 100.0


def get_is_h_bond_acceptor(atom: Chem.rdchem.Atom):
    """returns if an atom is a h bond acceptor, 1 for F, N and O that have 1 or more lone pairs"""
    if atom.GetSymbol() == "F" or atom.GetSymbol() == "O" or atom.GetSymbol() == "N":
        return 1
    else:
        return 0


def get_h_bond_acceptor(atom: Chem.rdchem.Atom):
    """returns if an atom is a h bond acceptor, 1 for F, N and O that have 1 or more lone pairs"""
    if atom.GetSymbol() == "N" and get_num_lone_pairs(atom) > 0:
        return 1
    elif atom.GetSymbol() == "O" and get_num_lone_pairs(atom) > 0:
        return 2
    elif atom.GetSymbol() == "F" and get_num_lone_pairs(atom) > 0:
        return 3
    else:
        return 0


def get_is_h_bond_donor(atom: Chem.rdchem.Atom):
    """returns if an atom is a h bond donor, 1 for N and O that have 1 or more H atoms"""
    if (atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0) or (atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0):
        return 1
    else:
        return 0


def get_h_bond_donor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
        return 1
    elif atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
        return 2
    elif atom.GetSymbol() == "F" and atom.GetTotalNumHs() > 0:
        return 3
    else:
        return 0


def get_in_ring_size(atom: Chem.rdchem.Atom):
    """returns the size of the ring the atom is in (the smallest one if more rings) /10.0 for scaling"""
    # for i in range(3, 10):
    #     if atom.IsInRingSize(i):
    #         #return float(i) / 10.0
    #         return float(i) / 10.0
    # else:
    #     return 0
    n = 0
    if atom.IsInRing():
        for i in range(1, 8):
            if atom.IsInRingSize(i):
                n=i
    else:
        n = 0
    return n


def get_num_lone_pairs(atom: Chem.rdchem.Atom):
    symbol = atom.GetSymbol()
    if symbol == 'C' or symbol == 'H':
        return 0 - atom.GetFormalCharge()
    elif symbol == 'S' or symbol == 'O':
        return 2 - atom.GetFormalCharge()
    elif symbol == 'N' or symbol == 'P':
        return 1 - atom.GetFormalCharge()
    elif symbol == 'F' or symbol == 'Cl' or symbol == 'Br' or symbol == 'I':
        return 3 - atom.GetFormalCharge()
    else:
        return 0


def get_vanderwaals_radius(atom: Chem.rdchem.Atom) -> float:
    symbol = atom.GetSymbol()
    if symbol == "H":
        return 1.1
    elif symbol == "C":
        return 1.7
    elif symbol == "B":
        return 1.92
    elif symbol == "N":
        return 1.55
    elif symbol == "O":
        return 1.52
    elif symbol == "F":
        return 1.47
    elif symbol == "Si":
        return 2.10
    elif symbol == "P":
        return 1.80
    elif symbol == "S":
        return 1.80
    elif symbol == "Cl":
        return 1.75
    elif symbol == "Br":
        return 1.85
    elif symbol == "I":
        return 1.98
    else:
        return 0.0


def get_electronegativity(atom: Chem.rdchem.Atom) -> float:
    symbol = atom.GetSymbol()
    if symbol == "H":
        return 2.20
    elif symbol == "Li":
        return 0.98
    elif symbol == "Be":
        return 1.57
    elif symbol == "C":
        return 2.55
    elif symbol == "B":
        return 2.04
    elif symbol == "N":
        return 3.04
    elif symbol == "O":
        return 3.44
    elif symbol == "F":
        return 3.98
    elif symbol == "Na":
        return 0.93
    elif symbol == "Mg":
        return 1.31
    elif symbol == "Al":
        return 1.61
    elif symbol == "Si":
        return 1.90
    elif symbol == "P":
        return 2.19
    elif symbol == "S":
        return 2.58
    elif symbol == "Cl":
        return 3.16
    elif symbol == "Br":
        return 2.96
    elif symbol == "I":
        return 2.66
    else:
        return 0.0


def get_is_isotope(atom: Chem.rdchem.Atom) -> int:
    isot = atom.GetIsotope()
    if isot > 0:
        return 1
    else:
        return 0


def get_atomic_number(atom: Chem.rdchem.Atom):
    nr = atom.GetAtomicNum()
    return nr


def get_if_bond_is_rotable(bond: Chem.rdchem.Bond):
    return 1 if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond.IsInRing()) else 0


def get_only_H(mol: Chem.rdchem.Mol):
    if mol.GetNumHeavyAtoms() == 0:
        return 1
    else:
        return 0


def get_sssr(mol: Chem.rdchem.Mol):
    return rdmolops.GetSSSR(mol)


def get_num_rotable_bonds(mol: Chem.rdchem.Mol):
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def get_num_aliphatic_rings(mol: Chem.rdchem.Mol):
    return rdMolDescriptors.CalcNumAliphaticRings(mol)


def get_num_aromatic_rings(mol: Chem.rdchem.Mol):
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def get_molar_mass(mol: Chem.rdchem.Mol):
    return Descriptors.MolWt(mol)


def get_carbon_flatness(mol: Chem.rdchem.Mol):
    count = 0.0
    total_count = 0.0
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'C':
            if a.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                count += 1.0
                total_count += 1.0
            else:
                total_count += 1.0
    if total_count == 0.0:
        total_count+=1.0
    return count/total_count


def get_flatness(mol: Chem.rdchem.Mol):
    count = 0.0
    total_count = 0.0
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'C' or a.GetSymbol() == 'N' or a.GetSymbol() == 'P' or a.GetSymbol() == 'Si':
            if a.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                count += 1.0
                total_count += 1.0
            else:
                total_count += 1.0
    if total_count == 0.0:
        total_count += 1.0

    return count/total_count


def get_aromatic_proportion(mol: Chem.rdchem.Mol):
    count = 0.0
    total_count = 0.0
    for a in mol.GetAtoms():
        if a.GetIsAromatic():
            count += 1.0
            total_count += 1.0
        else:
            total_count += 1.0
    return count/total_count
