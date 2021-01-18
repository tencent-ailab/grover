"""
Wrapper for multi-GPU training.
"""
# use Hovorod for multi-GPU pytorch training
try:
    import horovod.torch as mgw
    import torch

    print('using Horovod for multi-GPU training')
except ImportError:
    print('[WARNING] Horovod cannot be imported; multi-GPU training is unsupported')
    pass


class MultiGpuWrapper(object):
    """Wrapper for multi-GPU training."""

    def __init__(self):
        """Constructor function."""
        pass

    @classmethod
    def init(cls, *args):
        """Initialization."""

        try:
            return mgw.init(*args)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def size(cls, *args):
        """Get the number of workers at all nodes."""

        try:
            return mgw.size(*args)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def rank(cls, *args):
        """Get the rank of current worker at all nodes."""

        try:
            return mgw.rank(*args)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def local_size(cls, *args):
        """Get the number of workers at the current node."""

        try:
            return mgw.local_size(*args)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def local_rank(cls, *args):
        """Get the rank of current worker at the current node."""

        try:
            return mgw.local_rank(*args)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def DistributedOptimizer(cls, *args, **kwargs):
        """Get a distributed optimizer from the base optimizer."""

        try:
            return mgw.DistributedOptimizer(*args, **kwargs)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def broadcast_parameters(cls, *args, **kwargs):
        """Get a operation to broadcast all the parameters."""

        try:
            return mgw.broadcast_parameters(*args, **kwargs)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def broadcast_optimizer_state(cls, *args, **kwargs):
        """Get a operation to broadcast all the optimizer state."""

        try:
            return mgw.broadcast_optimizer_state(*args, **kwargs)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def broadcast(cls, *args, **kwargs):
        """Get a operation to broadcast all the optimizer state."""

        try:
            return mgw.broadcast(*args, **kwargs)
        except NameError:
            raise NameError('module <mgw> not imported')

    @classmethod
    def barrier(cls):
        """Add a barrier to synchronize different processes"""

        try:
            return mgw.allreduce(torch.tensor(0), name='barrier')
        except NameError:
            raise NameError('module <mgw> not imported')
