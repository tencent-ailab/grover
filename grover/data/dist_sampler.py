"""
The re-implemented distributed sampler for the distributed training of GROVER.
"""
import math
import time
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, sample_per_file=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.sample_per_file = sample_per_file
        self.shuffle = shuffle

    def get_indices(self):

        indices = list(range(len(self.dataset)))

        if self.sample_per_file is not None:
            indices = self.sub_indices_of_rank(indices)
        else:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            s = self.rank * self.num_samples
            e = min((self.rank + 1) * self.num_samples, len(indices))

            # indices = indices[self.rank:self.total_size:self.num_replicas]
            indices = indices[s:e]

        if self.shuffle:
            g = torch.Generator()
            # the seed need to be considered.
            g.manual_seed((self.epoch + 1) * (self.rank + 1) * time.time())
            idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx]

        # disable this since sub_indices_of_rank.
        # assert len(indices) == self.num_samples

        return indices

    def sub_indices_of_rank(self, indices):

        # fix generator for each epoch
        g = torch.Generator()
        # All data should be loaded in each epoch.
        g.manual_seed((self.epoch + 1) * 2 + 3)

        # the fake file indices to cache
        f_indices = list(range(int(math.ceil(len(indices) * 1.0 / self.sample_per_file))))
        idx = torch.randperm(len(f_indices), generator=g).tolist()
        f_indices = [f_indices[i] for i in idx]

        file_per_rank = int(math.ceil(len(f_indices) * 1.0 / self.num_replicas))
        # add extra fake file to make it evenly divisible
        f_indices += f_indices[:(file_per_rank * self.num_replicas - len(f_indices))]

        # divide index by rank
        rank_s = self.rank * file_per_rank
        rank_e = min((self.rank + 1) * file_per_rank, len(f_indices))

        # get file index for this rank
        f_indices = f_indices[rank_s:rank_e]
        # print("f_indices")
        # print(f_indices)
        res_indices = []
        for fi in f_indices:
            # get real indices for this rank
            si = fi * self.sample_per_file
            ei = min((fi + 1) * self.sample_per_file, len(indices))
            cur_idx = [indices[i] for i in range(si, ei)]
            res_indices += cur_idx

        self.num_samples = len(res_indices)
        return res_indices

    def __iter__(self):
        return iter(self.get_indices())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    # dataset = [1] * 9
    # ds = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)
    # print(ds.get_indices())
    # ds = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True)
    # print(ds.get_indices())

    dataset = [1] * 190001
    res = []
    ds = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, sample_per_file=777)
    res.extend(ds.get_indices())
    print(len(ds.get_indices()))
    ds = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, sample_per_file=777)
    res.extend(ds.get_indices())
    print(len(ds.get_indices()))
    print(len(set(res)))
    print("hello")
