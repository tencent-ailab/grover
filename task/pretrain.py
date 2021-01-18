"""
The GROVER pretrain function.
"""
import os
import time
from argparse import Namespace
from logging import Logger

import torch
from torch.utils.data import DataLoader

from grover.data.dist_sampler import DistributedSampler
from grover.data.groverdataset import get_data, split_data, GroverCollator, BatchMolDataset
from grover.data.torchvocab import MolVocab
from grover.model.models import GROVEREmbedding
from grover.util.multi_gpu_wrapper import MultiGpuWrapper as mgw
from grover.util.nn_utils import param_count
from grover.util.utils import build_optimizer, build_lr_scheduler
from task.grovertrainer import GROVERTrainer


def pretrain_model(args: Namespace, logger: Logger = None):
    """
    The entrey of pretrain.
    :param args: the argument.
    :param logger: the logger.
    :return:
    """

    # avoid auto optimized import by pycharm.
    a = MolVocab
    s_time = time.time()
    run_training(args=args, logger=logger)
    e_time = time.time()
    print("Total Time: %.3f" % (e_time - s_time))


def pre_load_data(dataset: BatchMolDataset, rank: int, num_replicas: int, sample_per_file: int = None, epoch: int = 0):
    """
    Pre-load data at the beginning of each epoch.
    :param dataset: the training dataset.
    :param rank: the rank of the current worker.
    :param num_replicas: the replicas.
    :param sample_per_file: the number of the data points in each file. When sample_per_file is None, all data will be
    loaded. It implies the testing phase. (TODO: bad design here.)
    :param epoch: the epoch number.
    :return:
    """
    mock_sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=False,
                                      sample_per_file=sample_per_file)
    mock_sampler.set_epoch(epoch)
    pre_indices = mock_sampler.get_indices()
    for i in pre_indices:
        dataset.load_data(i)


def run_training(args, logger):
    """
    Run the pretrain task.
    :param args:
    :param logger:
    :return:
    """

    # initalize the logger.
    if logger is not None:
        debug, _ = logger.debug, logger.info
    else:
        debug = print

    # initialize the horovod library
    if args.enable_multi_gpu:
        mgw.init()

    # binding training to GPUs.
    master_worker = (mgw.rank() == 0) if args.enable_multi_gpu else True
    # pin GPU to local rank. By default, we use gpu:0 for training.
    local_gpu_idx = mgw.local_rank() if args.enable_multi_gpu else 0
    with_cuda = args.cuda
    if with_cuda:
        torch.cuda.set_device(local_gpu_idx)

    # get rank an  number of workers
    rank = mgw.rank() if args.enable_multi_gpu else 0
    num_replicas = mgw.size() if args.enable_multi_gpu else 1
    # print("Rank: %d Rep: %d" % (rank, num_replicas))

    # load file paths of the data.
    if master_worker:
        print(args)
        if args.enable_multi_gpu:
            debug("Total workers: %d" % (mgw.size()))
        debug('Loading data')
    data, sample_per_file = get_data(data_path=args.data_path)

    # data splitting
    if master_worker:
        debug(f'Splitting data with seed 0.')
    train_data, test_data, _ = split_data(data=data, sizes=(0.9, 0.1, 0.0), seed=0, logger=logger)

    # Here the true train data size is the train_data divided by #GPUs
    if args.enable_multi_gpu:
        args.train_data_size = len(train_data) // mgw.size()
    else:
        args.train_data_size = len(train_data)
    if master_worker:
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(test_data):,}')

    # load atom and bond vocabulary and the semantic motif labels.
    atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)
    bond_vocab = MolVocab.load_vocab(args.bond_vocab_path)
    atom_vocab_size, bond_vocab_size = len(atom_vocab), len(bond_vocab)

    # Hard coding here, since we haven't load any data yet!
    fg_size = 85
    shared_dict = {}
    mol_collator = GroverCollator(shared_dict=shared_dict, atom_vocab=atom_vocab, bond_vocab=bond_vocab, args=args)
    if master_worker:
        debug("atom vocab size: %d, bond vocab size: %d, Number of FG tasks: %d" % (atom_vocab_size,
                                                                                    bond_vocab_size, fg_size))

    # Define the distributed sampler. If using the single card, the sampler will be None.
    train_sampler = None
    test_sampler = None
    shuffle = True
    if args.enable_multi_gpu:
        # If not shuffle, the performance may decayed.
        train_sampler = DistributedSampler(
            train_data, num_replicas=mgw.size(), rank=mgw.rank(), shuffle=True, sample_per_file=sample_per_file)
        # Here sample_per_file in test_sampler is None, indicating the test sampler would not divide the test samples by
        # rank. (TODO: bad design here.)
        test_sampler = DistributedSampler(
            test_data, num_replicas=mgw.size(), rank=mgw.rank(), shuffle=False)
        train_sampler.set_epoch(args.epochs)
        test_sampler.set_epoch(1)
        # if we enables multi_gpu training. shuffle should be disabled.
        shuffle = False

    # Pre load data. (Maybe unnecessary. )
    pre_load_data(train_data, rank, num_replicas, sample_per_file)
    pre_load_data(test_data, rank, num_replicas)
    if master_worker:
        # print("Pre-loaded training data: %d" % train_data.count_loaded_datapoints())
        print("Pre-loaded test data: %d" % test_data.count_loaded_datapoints())

    # Build dataloader
    train_data_dl = DataLoader(train_data,
                               batch_size=args.batch_size,
                               shuffle=shuffle,
                               num_workers=12,
                               sampler=train_sampler,
                               collate_fn=mol_collator)
    test_data_dl = DataLoader(test_data,
                              batch_size=args.batch_size,
                              shuffle=shuffle,
                              num_workers=10,
                              sampler=test_sampler,
                              collate_fn=mol_collator)

    # Build the embedding model.
    grover_model = GROVEREmbedding(args)

    #  Build the trainer.
    trainer = GROVERTrainer(args=args,
                            embedding_model=grover_model,
                            atom_vocab_size=atom_vocab_size,
                            bond_vocab_size=bond_vocab_size,
                            fg_szie=fg_size,
                            train_dataloader=train_data_dl,
                            test_dataloader=test_data_dl,
                            optimizer_builder=build_optimizer,
                            scheduler_builder=build_lr_scheduler,
                            logger=logger,
                            with_cuda=with_cuda,
                            enable_multi_gpu=args.enable_multi_gpu)

    # Restore the interrupted training.
    model_dir = os.path.join(args.save_dir, "model")
    resume_from_epoch = 0
    resume_scheduler_step = 0
    if master_worker:
        resume_from_epoch, resume_scheduler_step = trainer.restore(model_dir)
    if args.enable_multi_gpu:
        resume_from_epoch = mgw.broadcast(torch.tensor(resume_from_epoch), root_rank=0, name="resume_from_epoch").item()
        resume_scheduler_step = mgw.broadcast(torch.tensor(resume_scheduler_step),
                                              root_rank=0, name="resume_scheduler_step").item()
        trainer.scheduler.current_step = resume_scheduler_step
        print("Restored epoch: %d Restored scheduler step: %d" % (resume_from_epoch, trainer.scheduler.current_step))
    trainer.broadcast_parameters()

    # Print model details.
    if master_worker:
        # Change order here.
        print(grover_model)
        print("Total parameters: %d" % param_count(trainer.grover))

    # Perform training.
    for epoch in range(resume_from_epoch + 1, args.epochs):
        s_time = time.time()

        # Data pre-loading.
        if args.enable_multi_gpu:
            train_sampler.set_epoch(epoch)
            train_data.clean_cache()
            idxs = train_sampler.get_indices()
            for local_gpu_idx in idxs:
                train_data.load_data(local_gpu_idx)
        d_time = time.time() - s_time

        # perform training and validation.
        s_time = time.time()
        _, train_loss, _ = trainer.train(epoch)
        t_time = time.time() - s_time
        s_time = time.time()
        _, val_loss, detailed_loss_val = trainer.test(epoch)
        val_av_loss, val_bv_loss, val_fg_loss, _, _, _ = detailed_loss_val
        v_time = time.time() - s_time

        # print information.
        if master_worker:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  'loss_val_av: {:.6f}'.format(val_av_loss),
                  'loss_val_bv: {:.6f}'.format(val_bv_loss),
                  'loss_val_fg: {:.6f}'.format(val_fg_loss),
                  'cur_lr: {:.5f}'.format(trainer.scheduler.get_lr()[0]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time),
                  'd_time: {:.4f}s'.format(d_time), flush=True)

            if epoch % args.save_interval == 0:
                trainer.save(epoch, model_dir)


            trainer.save_tmp(epoch, model_dir, rank)

    # Only save final version.
    if master_worker:
        trainer.save(args.epochs, model_dir, "")
