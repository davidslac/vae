from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

class PartitionInfo(object):
    def __init__(self, **kwargs):
        fname = kwargs.pop('fname', None)
        info = kwargs.pop('info', None)
        if fname is not None and info is not None:
            return self.assign(fname, info)
        if (fname is not None) or (info is not None):
            raise Exception("if passing on of 'fname', 'info', must pass the other")

    def assign(self, fname, info):
        self.fname = fname
        self.weights = [elem[0] for elem in info]
        self.names = [elem[1] for elem in info]
        self.partitions = [elem[2] for elem in info]
        assert abs(sum(self.weights)-1.0)<1e-6, "weights don't sum to 1.0"
        self.num_samples = sum([len(elem) for elem in self.partitions])
        self.num_partitions = len(self.partitions)
        
    def save(self, h5out_fname):
        h5out = h5py.File(h5out_fname, 'w')
        h5out['orig_fname'] = self.fname
        part_group = h5out.create_group("partition")
        for partition in range(self.num_partitions):
            weight, name, samples = self.weights[partition], \
                                    self.names[partition], \
                                    self.partitions[partition]
            gr = part_group.create_group(name)
            gr['weight'] = weight
            gr['samples'] = samples
        h5out.close()

    def load(self, part_fname):
        h5 = h5py.File(part_fname, 'r')
        fname = h5['orig_fname'].value
        part_gr = h5['partition']
        self.names = part_gr.keys()
        self.names.sort()
        info = []
        for name in self.names:
            info.append((part_gr[name]['weight'].value,
                         name,
                         part_gr[name]['samples'][:]))
        self.assign(fname=fname, info=info)

    def get_sample2partition_array(self):
        arr = np.zeros(self.num_samples, np.int64)
        for num, partition in enumerate(self.partitions):
            arr[partition]=num
        return arr

