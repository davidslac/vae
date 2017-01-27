from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import collections
import h5py
from partitioninfo import PartitionInfo

def test_partition_info(fname):
    partition_info = PartitionInfo()
    partition_info.load(fname)
    sample2partition = partition_info.get_sample2partition_array()
    partitions = []
    for samples in batchgen.gen_samples(partition_info, epochs=1):
        partitions.extend(sample2partition[samples])
    partition_counts = collections.Counter(partitions)

    for partition,count in partition_counts.iteritems():
        print(" %d:%d" % (partition, count))

def get_samples(bins, partition_info, batch_size):
    random_values = np.random.rand(batch_size)
    partitions_to_sample_from = np.digitize(random_values, bins)-1
    list_of_samples_sets = partition_info.partitions
    batch_samples = []
    for partition, count in collections.Counter(partitions_to_sample_from).iteritems():
        num_samples = len(list_of_samples_sets[partition])
        idx = np.random.randint(0, num_samples, count)
        batch_samples.extend(list_of_samples_sets[partition][idx])
    return batch_samples

def gen_samples(partition_info, batch_size=32, epochs=0):
    '''generate samples per '''
    weights = [0] + partition_info.weights
    weights[-1] + 1e-6
    bins = np.cumsum(weights)
    total_samples = sum([len(sample_subset) for sample_subset in partition_info.partitions])
    
    epoch_number = 0
    samples_yieled_this_epoch = 0
    while True:
        if epochs > 0 and epoch_number == epochs:
            break
        yield get_samples(bins, partition_info, batch_size)
        samples_yieled_this_epoch += batch_size
        if samples_yieled_this_epoch > total_samples:
            epoch_number += 1
            samples_yieled_this_epoch = 0

def gen_batches(partition_fname, x_keys, y_dset2output, batch_size=32, epochs=0):
    partition_info = PartitionInfo()
    partition_info.load(partition_fname)
    gen = gen_samples(partition_info, batch_size=batch_size, epochs=epochs)
    h5=h5py.File(partition_info.fname,'r')
    NN = len(h5['imgs'])
    imgs = np.zeros((NN, 108, 54, 1), dtype=np.float32)
    imgs[:, 4:104, 2:52, 0] = h5['imgs'][:].reshape(NN, 100, 50)
    x = [imgs] #h5[ky][:] for ky in x_keys]
    y = {}
    for dset, output in y_dset2output.items():
        if dset in x_keys:
            y[output] = x[x_keys.index(dset)]
        else:
            y[output] = h5[dset][:]
    h5.close()

    for samples in gen:
        x_batch = [data[samples] for data in x]
        y_batch = {}
        for dset, output in y_dset2output.items():
            if dset in x_keys:
                y_batch[output] = x_batch[x_keys.index(dset)]
            else:
                y_batch[output] = y[output][samples]
        yield (x_batch, y_batch)

######## testing
def test_gen_samples(fname):
    partition_info = PartitionInfo()
    partition_info.load(fname)
    sample2partition = partition_info.get_sample2partition_array()
    partitions = []
    for samples in batchgen.gen_samples(partition_info, epochs=1):
        partitions.extend(sample2partition[samples])
    partition_counts = collections.Counter(partitions)

    for partition,count in partition_counts.iteritems():
        print(" %d:%d" % (partition, count))
