"""
The data splitting script for pretraining.
"""
import os
from argparse import ArgumentParser
import csv
import shutil
import numpy as np


import grover.util.utils as fea_utils


parser = ArgumentParser()
parser.add_argument("--data_path", default="../drug_data/grover_data/delaneyfreesolvlipo.csv")
parser.add_argument("--features_path", default="../drug_data/grover_data/delaneyfreesolvlipo_molbert.npz")
parser.add_argument("--sample_per_file", type=int, default=1000)
parser.add_argument("--output_path", default="../drug_data/grover_data/delaneyfreesolvlipo")


def load_smiles(data_path):
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        res = []
        for line in reader:
            res.append(line)
    return res, header


def load_features(data_path):
    fea = fea_utils.load_features(data_path)
    return fea


def save_smiles(data_path, index, data, header):
    fn = os.path.join(data_path, str(index) + ".csv")
    with open(fn, "w") as f:
        fw = csv.writer(f)
        fw.writerow(header)
        for d in data:
            fw.writerow(d)


def save_features(data_path, index, data):
    fn = os.path.join(data_path, str(index) + ".npz")
    np.savez_compressed(fn, features=data)


def run():
    args = parser.parse_args()
    res, header = load_smiles(data_path=args.data_path)
    fea = load_features(data_path=args.features_path)
    assert len(res) == fea.shape[0]

    n_graphs = len(res)
    perm = np.random.permutation(n_graphs)

    nfold = int(n_graphs / args.sample_per_file + 1)
    print("Number of files: %d" % nfold)
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)
    graph_path = os.path.join(args.output_path, "graph")
    fea_path = os.path.join(args.output_path, "feature")
    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(fea_path, exist_ok=True)

    for i in range(nfold):
        sidx = i * args.sample_per_file
        eidx = min((i + 1) * args.sample_per_file, n_graphs)
        indexes = perm[sidx:eidx]
        sres = [res[j] for j in indexes]
        sfea = fea[indexes]
        save_smiles(graph_path, i, sres, header)
        save_features(fea_path, i, sfea)

    summary_path = os.path.join(args.output_path, "summary.txt")
    summary_fout = open(summary_path, 'w')
    summary_fout.write("n_files:%d\n" % nfold)
    summary_fout.write("n_samples:%d\n" % n_graphs)
    summary_fout.write("sample_per_file:%d\n" % args.sample_per_file)
    summary_fout.close()


if __name__ == "__main__":
    run()
