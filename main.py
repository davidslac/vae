from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import h5py
import keras.backend as kb

from psmlearn.util import weights_from_edges
from psmlearn.pipeline import Pipeline
from psmlearn import h5util

import vae_lib
import vae_cnn_model
import batchgen
from partitioninfo import PartitionInfo

class MySteps(object):
    def __init__(self):
        self.data_dir = '/scratch/davidsch/dataprep'
        self.data_fnames = {'train': os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_train.h5'),
                            'validation': os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_validation.h5'),
                            'test': os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_validation.h5')
                            }
        self.split2pos = {'train': 0, 'validation': 1, 'test': 2}
        self.split2num_samples = {'train': 0, 'validation': 0, 'test': 0}
        for split in self.split2num_samples:
            h5 = h5py.File(self.data_fnames[split], 'r')
            self.split2num_samples[split] = len(h5['imgs'])
            h5.close()

    def init(self, config, pipeline):
        pass

    def balance_samples(self, config, pipeline, step2h5list, output_files):
        for idx, split_fname in enumerate(self.data_fnames.iteritems()):
            split, fname = split_fname
            vae_lib.balance(fname, output_files[self.split2pos[split]])

    def test_balance_samples(self, config, pipeline, step2h5list, output_files):
        fname = step2h5list['balance_samples'][0]
        batchgen.test_partition_info(fname)
        h5py.File(output_files[0], 'w')

    def edge_weights(self, config, pipeline, step2h5list, output_files):
        for idx, split_fname in enumerate(self.data_fnames.iteritems()):
            split, fname_in = split_fname
            h5in = h5py.File(fname_in, 'r')
            imgs = h5in['imgs'][:]
            num_images = len(imgs)
            weights = np.zeros((num_images, 100*50), np.uint8)
            for jdx in range(num_images):
                img = np.reshape(imgs[jdx, :], (100, 50))
                weights[jdx, :] = weights_from_edges(img).flatten()[:]
                if jdx % 1000 == 0:
                    print("%s: edge_weights jdx=%d" % (split, jdx))
            h5out = h5py.File(output_files[self.split2pos[split]], 'w')
            h5out['imgs'] = weights
            h5out.close()

    def view_edge_weights(self, plot, pipeline, plot_figh, config, step2h5list):
        split = 'train'
        img_fname = self.data_fnames[split]
        weight_fname = step2h5list['edge_weights'][self.split2pos[split]]
        imgs = h5py.File(img_fname, 'r')['imgs'][0:100]
        weights = h5py.File(weight_fname, 'r')['weights'][0:100]
        plt = pipeline.plt
        plt.figure(plot_figh)
        plt.clf()
        for img, weight in zip(imgs, weights):
            img = np.reshape(img, (100, 50))
            weight = np.reshape(weight, (100, 50))
            plt.subplot(1, 2, 1)
            plt.imshow(img, interpolation='none')
            plt.subplot(1, 2, 2)
            plt.imshow(weight, interpolation='none')
            plt.pause(1)

    def test_batch_read(self, config, pipeline, step2h5list, output_files):
        partition_fname = step2h5list['balance_samples'][self.split2pos['validation']]
        for x, y in batchgen.gen_batches(partition_fname,
                                         x_keys=['imgs'],
                                         y_dset2output={'imgs':'outA',
                                                        'gasdet':'outB',
                                                        'labels_onehot':'outC'}):
            print(x)
            print(y)
            break
        h5py.File(output_files[0],'w')
        
    def train(self, config, pipeline, step2h5list, output_files):
        t0 = time.time()
        ml = vae_cnn_model.Keras_VAE_CNN(batch_size=config.batch_size,
                                         latent_dim=config.latent_dim,
                                         optimizer=config.optimizer)
        ml.vae.summary()

        step2h5list['balance_samples'] = ['/scratch/davidsch/dataprep/vae_balance_samples_train.h5',
                                          '/scratch/davidsch/dataprep/vae_balance_samples_validation.h5',
                                          '/scratch/davidsch/dataprep/vae_balance_samples_test.h5']
        
        split2fit_info = {'train': {}, 'validation': {}, 'test': {}}
        for split, fit_info in split2fit_info.iteritems():
            partition_fname = step2h5list['balance_samples'][self.split2pos[split]]
            assert os.path.exists(partition_fname), "%s doesn't exist" % partition_fname
            gen = batchgen.gen_batches(partition_fname,
                                       x_keys=['imgs'],
                                       y_dset2output={'imgs': ml.img_out,
                                                      'gasdet':ml.gasdet_out,
                                                      'labels_onehot':ml.label_out},
                                       batch_size=config.batch_size)
            num_samples = self.split2num_samples[split]
            fit_info['gen'] = gen
            fit_info['num'] = num_samples

        train_split = 'train'
        hist = ml.vae.fit_generator(generator=split2fit_info[train_split]['gen'],
                                    samples_per_epoch=split2fit_info[train_split]['num'],
                                    nb_epoch=config.train_epochs,
                                    verbose=2,
                                    callbacks=None,
                                    pickle_safe=True,
                                    nb_worker=3,
                                    max_q_size=20,
                                    validation_data=split2fit_info['validation']['gen'],
                                    nb_val_samples=split2fit_info['validation']['num'])
        train_time = time.time()-t0
        ml.vae.save_weights(output_files[0])
        h5 = h5py.File(output_files[1],'w')
        h5['training_time']=train_time
        h5util.save_keras_hist(h5, hist)

        
def main(argv):
    pipeline = Pipeline(stepImpl=MySteps(),
                        session=kb.get_session(),
                        defprefix='vae',
                        outputdir='/scratch/davidsch/dataprep')

    pipeline.parser.add_argument('--batch_size', type=int, help='batch size', default=100)
    pipeline.parser.add_argument('--latent_dim', type=int, help='latent dim', default=2)
    pipeline.parser.add_argument('--train_epochs', type=int, help='number of training epochs', default=20)
    pipeline.parser.add_argument('--optimizer', type=str, help='optimizer', default='Adam')

#    pipeline.add_step_method(name='balance_samples', output_files=['_train', '_validation', '_test'])
#    pipeline.add_step_method(name='test_balance_samples')
#    pipeline.add_step_method(name='edge_weights', output_files=['_train', '_validation', '_test'])
#    pipeline.add_step_method_plot(name='view_edge_weights')
#    pipeline.add_step_method(name='test_batch_read')
    pipeline.add_step_method(name='train', output_files=['_model','_hist'])

    pipeline.run(argv[1:])
    
if __name__ == '__main__':
    main(sys.argv)
