"""
The fingerprint generation function.
"""
from argparse import Namespace
from logging import Logger
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from grover.data import MolCollator
from grover.data import MoleculeDataset
from grover.util.utils import get_data, create_logger, load_checkpoint


def do_generate(model: nn.Module,
                data: MoleculeDataset,
                args: Namespace,
                ) -> List[List[float]]:
    """
    Do the fingerprint generation on a dataset using the pre-trained models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param args: A StandardScaler object fit on the training targets.
    :return: A list of fingerprints.
    """
    model.eval()
    args.bond_drop_rate = 0
    preds = []

    mol_collator = MolCollator(args=args, shared_dict={})

    num_workers = 4
    mol_loader = DataLoader(data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=mol_collator)
    for item in mol_loader:
        _, batch, features_batch, _, _ = item
        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            preds.extend(batch_preds.data.cpu().numpy())
    return preds


def generate_fingerprints(args: Namespace, logger: Logger = None) -> List[List[float]]:
    """
    Generate the fingerprints.

    :param logger:
    :param args: Arguments.
    :return: A list of lists of target fingerprints.
    """

    checkpoint_path = args.checkpoint_paths[0]
    if logger is None:
        logger = create_logger('fingerprints', quiet=False)
    print('Loading data')
    test_data = get_data(path=args.data_path,
                         args=args,
                         use_compound_names=False,
                         max_data_size=float("inf"),
                         skip_invalid_smiles=False)
    test_data = MoleculeDataset(test_data)

    logger.info(f'Total size = {len(test_data):,}')
    logger.info(f'Generating...')
    # Load model
    model = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)
    model_preds = do_generate(
        model=model,
        data=test_data,
        args=args
    )

    return model_preds
