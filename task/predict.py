"""
The predict function using the finetuned model to make the prediction. .
"""
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from grover.data import MolCollator
from grover.data import MoleculeDataset
from grover.data import StandardScaler
from grover.util.utils import get_data, get_data_from_smiles, create_logger, load_args, get_task_names, tqdm, \
    load_checkpoint, load_scalars


def predict(model: nn.Module,
            data: MoleculeDataset,
            args: Namespace,
            batch_size: int,
            loss_func,
            logger,
            shared_dict,
            scaler: StandardScaler = None
            ) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    # debug = logger.debug if logger is not None else print
    model.eval()
    args.bond_drop_rate = 0
    preds = []

    # num_iters, iter_step = len(data), batch_size
    loss_sum, iter_count = 0, 0

    mol_collator = MolCollator(args=args, shared_dict=shared_dict)
    # mol_dataset = MoleculeDataset(data)

    num_workers = 4
    mol_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=mol_collator)
    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        class_weights = torch.ones(targets.shape)
        if next(model.parameters()).is_cuda:
            targets = targets.cuda()
            mask = mask.cuda()
            class_weights = class_weights.cuda()
        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            iter_count += 1
            if args.fingerprint:
                preds.extend(batch_preds.data.cpu().numpy())
                continue

            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss_sum += loss.item()
        # Collect vectors
        batch_preds = batch_preds.data.cpu().numpy().tolist()
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        preds.extend(batch_preds)

    loss_avg = loss_sum / iter_count
    return preds, loss_avg


def make_predictions(args: Namespace, newest_train_args=None, smiles: List[str] = None):
    """
    Makes predictions. If smiles is provided, makes predictions on smiles.
    Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')

    path = args.checkpoint_paths[0]
    scaler, features_scaler = load_scalars(path)
    train_args = load_args(path)

    # Update args with training arguments saved in checkpoint
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # update args with newest training args
    if newest_train_args is not None:
        for key, value in vars(newest_train_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)


    # deal with multiprocess problem
    args.debug = True

    logger = create_logger('predict', quiet=False)
    print('Loading data')
    args.task_names = get_task_names(args.data_path)
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.data_path, args=args,
                             use_compound_names=args.use_compound_names, skip_invalid_smiles=False)


    args.num_tasks = test_data.num_tasks()
    args.features_size = test_data.features_size()

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data))]
    full_data = test_data
    # test_data = MoleculeDataset([test_data[i] for i in valid_indices])
    test_data_list = []
    for i in valid_indices:
        test_data_list.append(test_data[i])
    test_data = MoleculeDataset(test_data_list)

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if hasattr(train_args, 'features_scaling'):
        if train_args.features_scaling:
            test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if hasattr(args, 'num_tasks'):
        sum_preds = np.zeros((len(test_data), args.num_tasks))
    print(f'Predicting...')
    shared_dict = {}
    # loss_func = torch.nn.BCEWithLogitsLoss()
    count = 0
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)
        model_preds, _ = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler,
            shared_dict=shared_dict,
            args=args,
            logger=logger,
            loss_func=None
        )

        if args.fingerprint:
            return model_preds

        sum_preds += np.array(model_preds, dtype=float)
        count += 1

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)

    # Save predictions
    assert len(test_data) == len(avg_preds)

    # Put Nones for invalid smiles
    args.valid_indices = valid_indices
    avg_preds = np.array(avg_preds)
    test_smiles = full_data.smiles()
    return avg_preds, test_smiles


def write_prediction(avg_preds, test_smiles, args):
    """
    write prediction to disk
    :param avg_preds: prediction value
    :param test_smiles: input smiles
    :param args: Arguments
    """
    if args.dataset_type == 'multiclass':
        avg_preds = np.argmax(avg_preds, -1)
    full_preds = [[None]] * len(test_smiles)
    for i, si in enumerate(args.valid_indices):
        full_preds[si] = avg_preds[i]
    result = pd.DataFrame(data=full_preds, index=test_smiles, columns=args.task_names)
    result.to_csv(args.output_path)
    print(f'Saving predictions to {args.output_path}')



def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func,
                         dataset_type: str,
                         logger = None) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    if dataset_type == 'multiclass':
        results = metric_func(np.argmax(preds, -1), [i[0] for i in targets])
        return [results]

    # info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                # info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                # info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func,
             loss_func,
             batch_size: int,
             dataset_type: str,
             args: Namespace,
             shared_dict,
             scaler: StandardScaler = None,
             logger = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds, loss_avg = predict(
        model=model,
        data=data,
        loss_func=loss_func,
        batch_size=batch_size,
        scaler=scaler,
        shared_dict=shared_dict,
        logger=logger,
        args=args
    )

    targets = data.targets()
    if scaler is not None:
        targets = scaler.inverse_transform(targets)



    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results, loss_avg
