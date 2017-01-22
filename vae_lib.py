from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
from scipy.misc import imresize
import collections

from partitioninfo import PartitionInfo, test_partition_info

'''
Balances the samples.
This balances for the VAE model.
'''

def make_reduced_images(flattened_images, orig_dim, reduced_dim):
    assert len(reduced_dim) == 2
    assert len(orig_dim) == 2
    assert len(flattened_images.shape) == 2
    assert flattened_images.shape[1] == orig_dim[0]*orig_dim[1]
    
    num_images = len(flattened_images)
    all_reduced_shape = (num_images, reduced_dim[0] * reduced_dim[1])
    reduced_images = np.zeros(all_reduced_shape, flattened_images.dtype)
    for idx in range(num_images):
        orig = np.reshape(flattened_images[idx],orig_dim)
        reduced = imresize(orig, reduced_dim)
        reduced_images[idx, :] = reduced.flatten()[:]
    return reduced_images


def get_hash_images(flattened_images, orig_dim):
    reduced_dim = (4, 4)
    flattened_images = make_reduced_images(flattened_images, orig_dim, reduced_dim)
    grand_mean = np.mean(flattened_images, axis=0)
    assert grand_mean.shape == (flattened_images.shape[1],)
    image_bits = np.packbits(flattened_images > grand_mean, axis=1)
    assert len(image_bits) == len(flattened_images)
    assert image_bits.shape[1] == 2
    image_keys = image_bits[:, 0].astype(np.int64)
    for col in range(1, 2):
        image_keys = np.left_shift(image_keys, 8)
        image_keys += image_bits[:, col]
    return image_keys
    

def div2(data):
    data = np.sort(data)
    num_elem = len(data)
    assert num_elem > 4
    p5 = min(num_elem-1, max(0, int(round(0.5*num_elem))))
    p95 = min(num_elem-1, max(0, int(round(0.95*num_elem))))
    split_at = (data[p5] + data[p95]) / 2.0
    return split_at


def check_is_partition(partition, total):
    present = np.zeros(total, np.int8)
    for subset in partition:
        present[subset]=1
    num_in_partitions = np.sum(present)
    assert num_in_partitions == total, "set of %d is not a partition, only contains %d unique members of total == %d" % \
        (len(partition), num_in_partitions, total)


def int_split(data, data_orig_rows, number_of_partitions):
    assert len(data) == len(data_orig_rows)
    unique_elem = set(list(data))
    assert len(unique_elem) > 2 * number_of_partitions, "Want # unique values to be at least 2*%d since num=%d, but it is %d" % \
                                                        (number_of_partitions, number_of_partitions, len(unique_elem))
    sorted_unique_elem = np.sort(np.array(list(unique_elem)))
    split_samples = []
    idx_a = 0
    for part in range(1, number_of_partitions+1):
        if part == number_of_partitions:
            idx_b = len(sorted_unique_elem)-1
        else:
            idx_b = max(0, min(len(sorted_unique_elem) - 1, int(round((part*len(sorted_unique_elem)) / float(number_of_partitions)))))
        assert idx_b > idx_a
        val_a = sorted_unique_elem[idx_a]
        val_b = sorted_unique_elem[idx_b]
        assert val_b > val_a
        if part == number_of_partitions:
            log_idx = np.logical_and(data >= val_a, data <= val_b)
        else:
            log_idx = np.logical_and(data >= val_a, data < val_b)
        split_samples.append(list(data_orig_rows[log_idx]))
        idx_a = idx_b
    return split_samples


def balance_images(samples, all_images, num_partitions=4, orig_img_size=(100, 50)):
    images = all_images[list(samples)]
    img_keys = get_hash_images(images, orig_img_size)
    new_samples = int_split(img_keys, samples, num_partitions)
    new_dict = {}
    for idx, subset in enumerate(new_samples):
        new_dict['img_%d' % idx] = subset
    orig_set = set(samples)
    new_set = set([])
    for subset in new_samples:
        new_set = new_set.union(set(subset))
    assert new_set == orig_set, "error paritioning images, num_samples=%d " + \
        "new_set has %d, and orig_set has %d" % \
        (len(samples), len(new_set), len(orig_set))

    return new_dict


def balance(h5in_fname, h5out_fname):
    h5in = h5py.File(h5in_fname, 'r')
    imgs = h5in['imgs'][:]
    labels = h5in['labels'][:]
    gasdet = np.mean(h5in['gasdet'][:], axis=1)
    h5in.close()

    num_samples = len(imgs)
    all_samples = np.arange(num_samples)
    assert len(gasdet) == num_samples
    gasdet_split_value = div2(gasdet)
    
    subsets = {'label_0': all_samples[labels == 0],
               'label_1':  {}
               }
    subsets['label_1']['gasdet_low'] = all_samples[np.logical_and(labels == 1, gasdet >= gasdet_split_value)]
    subsets['label_1']['gasdet_high'] = all_samples[np.logical_and(labels == 1, gasdet < gasdet_split_value)]
    check_is_partition([subsets['label_0'],
                        subsets['label_1']['gasdet_low'],
                        subsets['label_1']['gasdet_high']],
                       num_samples)

    subsets['label_0'] = balance_images(subsets['label_0'], imgs, num_partitions=4, orig_img_size=(100, 50))
    subsets['label_1']['gasdet_low'] = balance_images(subsets['label_1']['gasdet_low'], imgs,
                                                      num_partitions=4, orig_img_size=(100, 50))
    subsets['label_1']['gasdet_high'] = balance_images(subsets['label_1']['gasdet_high'], imgs,
                                                       num_partitions=4, orig_img_size=(100, 50))
    
    w1 = (1/2)*(1/4)
    w2 = (1/2)*(1/2)*(1/4)
    info=[(w1, 'label_0_img_0', subsets['label_0']['img_0']),
          (w1, 'label_0_img_1', subsets['label_0']['img_1']),
          (w1, 'label_0_img_2', subsets['label_0']['img_2']),
          (w1, 'label_0_img_3', subsets['label_0']['img_3']),
          (w2, 'label_1_gasdet_low_img_0', subsets['label_1']['gasdet_low']['img_0']),
          (w2, 'label_1_gasdet_low_img_1', subsets['label_1']['gasdet_low']['img_1']),
          (w2, 'label_1_gasdet_low_img_2', subsets['label_1']['gasdet_low']['img_2']),
          (w2, 'label_1_gasdet_low_img_3', subsets['label_1']['gasdet_low']['img_3']),
          (w2, 'label_1_gasdet_high_img_0', subsets['label_1']['gasdet_high']['img_0']),
          (w2, 'label_1_gasdet_high_img_1', subsets['label_1']['gasdet_high']['img_1']),
          (w2, 'label_1_gasdet_high_img_2', subsets['label_1']['gasdet_high']['img_2']),
          (w2, 'label_1_gasdet_high_img_3', subsets['label_1']['gasdet_high']['img_3'])]
    partition_info = PartitionInfo(fname=h5in_fname, info=info)
    partition_info.save(h5out_fname)


