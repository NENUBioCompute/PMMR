import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType
import pickle
import torch
from rdkit.Chem import AllChem, rdMolDescriptors
# from collections import defaultdict
# from deepchem.feat import SmilesTokenizer
from transformers import BertTokenizer, AutoTokenizer,BertModel,RobertaModel,RobertaTokenizer
import os

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    features = []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    adj = adjacent_matrix(mol)

    return features, adj

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])

def get_mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError("SMILES cannot been parsed!")

    features, adj = smile_to_graph(smiles)

    features = np.array(features)


    return features,adj




def get_smiles_embedding(smiles):

    # vocab_path = './vocab.txt'  # path to vocab.txt
    # tokenizer_ = SmilesTokenizer(vocab_path)
    tokenizer_ = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model_ = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model_.eval()
    with torch.no_grad():
        outputs_ = model_(**tokenizer_(smiles, return_tensors='pt'))


    return outputs_.last_hidden_state[0][1:outputs_.last_hidden_state.shape[1]-1]


