"""
Computes and saves molecular features for a dataset.
"""
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import List, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from grover.util.utils import get_data, makedirs, load_features, save_features
from grover.data.molfeaturegenerator import get_available_features_generators, \
    get_features_generator
from grover.data.task_labels import rdkit_functional_group_label_features_generator



def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .npz files in load_dir.

    Assumes temporary files are named in order 0.npz, 1.npz, ...

    :param temp_dir: Directory in which temporary .npz files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def generate_and_save_features(args: Namespace):
    """
    Computes and saves features for a dataset of molecules as a 2D array in a .npz file.

    :param args: Arguments.
    """
    # Create directory for save_path
    makedirs(args.save_path, isfile=True)

    # Get data and features function
    data = get_data(path=args.data_path, max_data_size=None)
    features_generator = get_features_generator(args.features_generator)
    temp_save_dir = args.save_path + '_temp'

    # Load partially complete data
    if args.restart:
        if os.path.exists(args.save_path):
            os.remove(args.save_path)
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(args.save_path):
            raise ValueError(f'"{args.save_path}" already exists and args.restart is False.')

        if os.path.exists(temp_save_dir):
            features, temp_num = load_temp(temp_save_dir)

    if not os.path.exists(temp_save_dir):
        makedirs(temp_save_dir)
        features, temp_num = [], 0

    # Build features map function
    data = data[len(features):]  # restrict to data for which features have not been computed yet
    mols = (d.smiles for d in data)

    if args.sequential:
        features_map = map(features_generator, mols)
    else:
        features_map = Pool(30).imap(features_generator, mols)

    # Get features
    temp_features = []
    for i, feats in tqdm(enumerate(features_map), total=len(data)):
        temp_features.append(feats)

        # Save temporary features every save_frequency
        if (i > 0 and (i + 1) % args.save_frequency == 0) or i == len(data) - 1:
            save_features(os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features)
            features.extend(temp_features)
            temp_features = []
            temp_num += 1

    try:
        # Save all features
        save_features(args.save_path, features)

        # Remove temporary features
        shutil.rmtree(temp_save_dir)
    except OverflowError:
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--features_generator', type=str, required=True,
                        choices=get_available_features_generators(),
                        help='Type of features to generate')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to .npz file where features will be saved as a compressed numpy archive')
    parser.add_argument('--save_frequency', type=int, default=10000,
                        help='Frequency with which to save the features')
    parser.add_argument('--restart', action='store_true', default=False,
                        help='Whether to not load partially complete featurization and instead start from scratch')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to task sequentially rather than in parallel')
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = args.data_path.split('csv')[0] + 'npz'
    generate_and_save_features(args)
