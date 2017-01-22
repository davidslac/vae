from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import h5py
import keras.backend as K

from psmlearn.util import weights_from_edges
from psmlearn.pipeline import Pipeline

import vae_lib
import batchgen

class MySteps(object):
    def __init__(self):
        self.data_dir = '/scratch/davidsch/dataprep'
        self.train_fname = os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_train.h5')
        self.validation_fname = os.path.join(self.data_dir, 'xtcav_autoencoder_dense_prep_validation.h5')

    def add_commandline_arguments(self, parser):
        pass

    def init(self, config, pipeline):
        pass

    def balance_samples(self, config, pipeline, step2h5list, output_files):
        vae_lib.balance(self.train_fname, output_files[0])
#        vae_lib.balance(self.validation_fname, output_files[0])

    def test_batch_read(self, config, pipeline, step2h5list, output_files):
        fname = step2h5list['balance_samples'][0]
        vae_lib.test_partition_info(fname)
        h5py.File(output_files[0], 'w')

    def edge_weights(self, config, pipeline, step2h5list, output_files):
        for split, fname_in, fname_out in zip(['train','validation'],
                                              [self.train_fname, self.validation_fname],
                                              output_files):
            h5in = h5py.File(fname_in,'r')
            imgs = h5in['imgs'][:]
            num_images = len(imgs)
            weights = np.zeros((num_images, 100*50), np.uint8)
            for idx in range(num_images):
                img = np.reshape(imgs[idx,:], (100,50))
                weights[idx,:] = weights_from_edges(img).flatten()[:]
                if idx % 1000 == 0: print("%s: edge_weights idx=%d" % (split, idx))
            h5out = h5py.File(fname_out, 'w')
            h5out['weights'] = weights
            h5out.close()

    def view_edge_weights(self, plot, pipeline, plotFigH, config, step2h5list):
        imgs = h5py.File(self.train_fname,'r')['imgs'][0:100]
        weights = h5py.File(step2h5list['edge_weights'][0],'r')['weights'][0:100]
        plt = pipeline.plt
        plt.figure(plotFigH)
        plt.clf()
        for img, weight in zip(imgs,weights):
            img = np.reshape(img,(100,50))
            weight = np.reshape(weight,(100,50))
            plt.subplot(1,2,1)
            plt.imshow(img, interpolation='none')
            plt.subplot(1,2,2)
            plt.imshow(weight, interpolation='none')
            plt.pause(1)

def main(argv):
    my_steps = MySteps()
    pipeline = Pipeline(stepImpl=my_steps,
                        session=K.get_session(),
                        defprefix='vae',
                        outputdir='/scratch/davidsch/dataprep')

    my_steps.add_commandline_arguments(pipeline.parser)
    pipeline.add_step_method(name='balance_samples')
#    pipeline.add_step_method(name='test_batch_read')
    pipeline.add_step_method(name='edge_weights', output_files=['_train','_validation'])
    pipeline.add_step_method_plot(name='view_edge_weights')

    pipeline.run(argv[1:])
    
if __name__ == '__main__':
    main(sys.argv)
