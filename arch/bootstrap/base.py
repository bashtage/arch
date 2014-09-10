import numpy as np
from numpy.random import RandomState

from arch.compat.python import iteritems, itervalues, add_metaclass
from arch.utils import DocStringInheritor


@add_metaclass(DocStringInheritor)
class IIDBootstrap(object):

    def __init__(self, *args, **kwargs):
        self.iterations = 0
        self._iter_count = 0
        self.random_state = RandomState()
        self.initial_state = self.random_state.get_state()
        self.data = args
        self._args = args
        self._kwargs = kwargs
        if args:
            self._num_items = len(args[0])
        elif kwargs:
            self._num_items = len(kwargs[kwargs.keys()[0]])

        all_args = list(args)
        all_args.extend([v for v in itervalues(kwargs)])

        for arg in all_args:
            if len(arg) != self._num_items:
                raise ValueError("All inputs must have the same number of "
                                 "elements in axis 0")

        self.pos_data = args
        self.named_data = kwargs

        self.indices = np.arange(self._num_items)
        for key, value in iteritems(kwargs):
            self.__setattr__(key, value)

    def get_state(self):
        return self.random_state.get_state()

    def set_state(self):
        return self.random_state.set_state()

    def seed(self, value):
        self.random_state.seed(value)
        return None

    def reset(self):
        self.random_state.set_state(self.initial_state)

    def bootstrap(self, num):
        self.iterations = num

    def __iter__(self):
        for i in range(self.num):
            indices = self.update_indices()
            yield self._resample(indices)

    def update_indices(self):
        raise NotImplementedError("Subclasses must implement")

    def _resample(self, indices):
        pos_data = [arg[indices] for arg in self._args]
        named_data = {k: v[indices] for k,v in iteritems(self._kwargs)}
        self.pos_data = pos_data
        self.named_data = named_data
        self.data = (pos_data, named_data)
        return self.data

class CircularBlockBootstrap(IIDBootstrap):
    def __init__(self, block_size, *args, **kwargs):
        super(IIDBootstrap, self).__init__(*args, **kwargs)
        self.block_size = block_size

    def update_indices(self):
        raise NotImplementedError

class StationaryBootstrap(IIDBootstrap):
    def __init__(self, block_size, *args, **kwargs):
        super(IIDBootstrap, self).__init__()
        self.block_size = block_size
        self._p = 1.0 / block_size

    def update_indices(self):
        raise NotImplementedError

class MovingBlockBootstrap(IIDBootstrap):
    def __init__(self, block_size, *args, **kwargs):
        super(IIDBootstrap, self).__init__(*args, **kwargs)
        self.block_size = block_size

    def update_indices(self):
        raise NotImplementedError


class MOONBootstrap(IIDBootstrap):
    def __init__(self, block_size, *args, **kwargs):
        super(IIDBootstrap, self).__init__(*args, **kwargs)
        self.block_size = block_size

    def update_indices(self):
        raise NotImplementedError

