"""
The vocabulary building scripts.
"""
import os

from grover.data.torchvocab import MolVocab


def build():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../../dataset/grover_new_dataset/druglike_merged_refine2.csv", type=str)
    parser.add_argument('--vocab_save_folder', default="../../dataset/grover_new_dataset", type=str)
    parser.add_argument('--dataset_name', type=str, default=None,
                        help="Will be the first part of the vocab file name. If it is None,"
                             "the vocab files will be: atom_vocab.pkl and bond_vocab.pkl")
    parser.add_argument('--vocab_max_size', type=int, default=None)
    parser.add_argument('--vocab_min_freq', type=int, default=1)
    args = parser.parse_args()

    # fin = open(args.data_path, 'r')
    # lines = fin.readlines()

    for vocab_type in ['atom', 'bond']:
        vocab_file = f"{vocab_type}_vocab.pkl"
        if args.dataset_name is not None:
            vocab_file = args.dataset_name + '_' + vocab_file
        vocab_save_path = os.path.join(args.vocab_save_folder, vocab_file)

        os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
        vocab = MolVocab(file_path=args.data_path,
                         max_size=args.vocab_max_size,
                         min_freq=args.vocab_min_freq,
                         num_workers=100,
                         vocab_type=vocab_type)
        print(f"{vocab_type} vocab size", len(vocab))
        vocab.save_vocab(vocab_save_path)


if __name__ == '__main__':
    build()
