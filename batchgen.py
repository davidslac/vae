from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import collections

def get_samples(bins, partition_info, batchsize):
    random_values = np.random.rand(batchsize)
    partitions_to_sample_from = np.digitize(random_values, bins)-1
    list_of_samples_sets = partition_info.partitions
    batch_samples = []
    for partition, count in collections.Counter(partitions_to_sample_from).iteritems():
        num_samples = len(list_of_samples_sets[partition])
        idx = np.random.randint(0, num_samples, count)
        batch_samples.extend(list_of_samples_sets[partition][idx])
    return batch_samples

def gen_samples(partition_info, batchsize=32, epochs=0):
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
        yield get_samples(bins, partition_info, batchsize)
        samples_yieled_this_epoch += batchsize
        if samples_yieled_this_epoch > total_samples:
            epoch_number += 1
            samples_yieled_this_epoch = 0
