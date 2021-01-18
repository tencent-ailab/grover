"""
The dataset used in training GROVER.
"""
import math
import os
import csv
from typing import Union, List
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem

import grover.util.utils as feautils
from grover.data import mol2graph
from grover.data.moldataset import MoleculeDatapoint
from grover.data.task_labels import atom_to_vocab, bond_to_vocab


def get_data(data_path, logger=None):
    """
    Load data from the data_path.
    :param data_path: the data_path.
    :param logger: the logger.
    :return:
    """
    debug = logger.debug if logger is not None else print
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    feature_path = os.path.join(data_path, "feature")

    fin = open(summary_path)
    n_files = int(fin.readline().strip().split(":")[-1])
    n_samples = int(fin.readline().strip().split(":")[-1])
    sample_per_file = int(fin.readline().strip().split(":")[-1])
    debug("Loading data:")
    debug("Number of files: %d" % n_files)
    debug("Number of samples: %d" % n_samples)
    debug("Samples/file: %d" % sample_per_file)

    datapoints = []
    for i in range(n_files):
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        feature_path_i = os.path.join(feature_path, str(i) + ".npz")
        n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        datapoints.append(BatchDatapoint(smiles_path_i, feature_path_i, n_samples_i))
    return BatchMolDataset(datapoints), sample_per_file


def split_data(data,
               split_type='random',
               sizes=(0.8, 0.1, 0.1),
               seed=0,
               logger=None):
    """
    Split data with given train/validation/test ratio.
    :param data:
    :param split_type:
    :param sizes:
    :param seed:
    :param logger:
    :return:
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if split_type == "random":
        data.shuffle(seed=seed)
        data = data.data

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return BatchMolDataset(train), BatchMolDataset(val), BatchMolDataset(test)
    else:
        raise NotImplementedError("Do not support %s splits" % split_type)


class BatchDatapoint:
    def __init__(self,
                 smiles_file,
                 feature_file,
                 n_samples,
                 ):
        self.smiles_file = smiles_file
        self.feature_file = feature_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.datapoints = None

    def load_datapoints(self):
        features = self.load_feature()
        self.datapoints = []

        with open(self.smiles_file) as f:
            reader = csv.reader(f)
            next(reader)
            for i, line in enumerate(reader):
                # line = line[0]
                d = MoleculeDatapoint(line=line,
                                      features=features[i])
                self.datapoints.append(d)

        assert len(self.datapoints) == self.n_samples

    def load_feature(self):
        return feautils.load_features(self.feature_file)

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        assert self.datapoints is not None
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None


class BatchMolDataset(Dataset):
    def __init__(self, data: List[BatchDatapoint],
                 graph_per_file=None):
        self.data = data

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        # print(idx)
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        return self.data[dp_idx][real_idx]

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        if not self.data[dp_idx].is_loaded():
            self.data[dp_idx].load_datapoints()

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res


class GroverCollator(object):
    def __init__(self, shared_dict, atom_vocab, bond_vocab, args):
        self.args = args
        self.shared_dict = shared_dict
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab

    def atom_random_mask(self, smiles_batch):
        """
        Perform the random mask operation on atoms.
        :param smiles_batch:
        :return: The corresponding atom labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            mlabel = [0] * mol.GetNumAtoms()
            n_mask = math.ceil(mol.GetNumAtoms() * percent)
            perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
            for p in perm:
                atom = mol.GetAtomWithIdx(int(p))
                mlabel[p] = self.atom_vocab.stoi.get(atom_to_vocab(mol, atom), self.atom_vocab.other_index)

            vocab_label.extend(mlabel)
        return vocab_label

    def bond_random_mask(self, smiles_batch):
        """
        Perform the random mask operaiion on bonds.
        :param smiles_batch:
        :return: The corresponding bond labels.
        """
        # There is a zero padding.
        vocab_label = [0]
        percent = 0.15
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            nm_atoms = mol.GetNumAtoms()
            nm_bonds = mol.GetNumBonds()
            mlabel = []
            n_mask = math.ceil(nm_bonds * percent)
            perm = np.random.permutation(nm_bonds)[:n_mask]
            virtual_bond_id = 0
            for a1 in range(nm_atoms):
                for a2 in range(a1 + 1, nm_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue
                    if virtual_bond_id in perm:
                        label = self.bond_vocab.stoi.get(bond_to_vocab(mol, bond), self.bond_vocab.other_index)
                        mlabel.extend([label])
                    else:
                        mlabel.extend([0])

                    virtual_bond_id += 1
            # todo: might need to consider bond_drop_rate
            # todo: double check reverse bond
            vocab_label.extend(mlabel)
        return vocab_label

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        atom_vocab_label = torch.Tensor(self.atom_random_mask(smiles_batch)).long()
        bond_vocab_label = torch.Tensor(self.bond_random_mask(smiles_batch)).long()
        fgroup_label = torch.Tensor([d.features for d in batch]).float()
        # may be some mask here
        res = {"graph_input": batchgraph,
               "targets": {"av_task": atom_vocab_label,
                           "bv_task": bond_vocab_label,
                           "fg_task": fgroup_label}
               }
        return res
