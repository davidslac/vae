from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# external
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Convolution2D, Flatten, Reshape
from keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D
from keras import objectives
import keras.backend as kb


def sampling(args, **kwargs):
    z_mean, z_log_var = args
    batch_size = kwargs.pop('batch_size')
    latent_dim = kwargs.pop('latent_dim')
    epsilon_std = kwargs.pop('epsilon_std')
    epsilon = kb.random_normal(shape=(batch_size, latent_dim),
                               mean=0., std=epsilon_std)
    return z_mean + kb.exp(z_log_var / 2.0) * epsilon


class Keras_VAE_CNN(object):
    '''defines VAE
    usage
    ML = Keras_VAE_CNN(batch_size, latent_dim)
    ML.
    '''
    def __init__(self, batch_size, latent_dim, optimizer):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epsilon_std = 1.0
        self.img_H = 108
        self.img_W = 54

        self.input_img = Input(batch_shape=(batch_size, self.img_H, self.img_W, 1))
        self.input_labels = Input(batch_shape=(batch_size,2))
        self.input_gasdet = Input(batch_shape=(batch_size,1))

        self.hidden = None
        self.z_mean = None
        self.z_log_var = None
        self.z = None
        self.encoder_layers = None
        
        # define names for model outputs
        self.img_out = 'img_decoded_mean'
        self.gasdet_out = 'gasdet'
        self.label_out = 'label_logits'
        self.img_part_of_img_loss = 'img_part'
        self.k1_part_of_img_loss = 'k1_part'
        self.outputs = {self.img_out:None, self.gasdet_out:None, self.label_out:None}
        self.metrics = {self.img_out:'mean_squared_error', 
                        self.gasdet_out:'mean_squared_error', 
                        self.label_out:'binary_accuracy'}

        self.k1_loss_weight = 1.0
        self.loss_weights = [1.0, 1.0, 1.0]
        self.loss_fns = {self.img_out:None, self.gasdet_out:None, self.label_out:None}
        self.loss_results = {self.img_out:None, self.gasdet_out:None, self.label_out:None,
                             self.img_part_of_img_loss:None,
                             self.k1_part_of_img_loss:None}

        self.vae = None
        self.encoder = None
        self.decoder_input = None
        self.generator = None

        self.make_img_output_and_primary_ops()
        self.make_auxillary_outputs()
        self.define_models_from_outputs()
        self.make_losses()
        self.compile_vae(optimizer)

    def make_img_output_and_primary_ops(self):
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(self.input_img)
        # (108, 54, 16)

        x = MaxPooling2D((2, 2), border_mode='same')(x)
        # (54, 27, 16)

        x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
        # (54, 27, 8)

        x = MaxPooling2D((3, 3), border_mode='same')(x)
        # (18, 9, 8)

        x = Convolution2D(4, 7, 5, activation='relu', border_mode='same')(x)
        # (18, 9, 4 )

        x = MaxPooling2D((3, 3), border_mode='same')(x)
        # (6, 3, 4)

        self.hidden = Flatten()(x)

        self.z_mean = Dense(self.latent_dim)(self.hidden)
        self.z_log_var = Dense(self.latent_dim)(self.hidden)

        self.z = Lambda(sampling,
                        arguments={'batch_size': self.batch_size,
                                   'latent_dim': self.latent_dim,
                                   'epsilon_std': self.epsilon_std})([self.z_mean, self.z_log_var])

        self.encoder_layers = []
        self.encoder_layers.append(Dense(6 * 3 * 4, activation='relu'))
        self.encoder_layers.append(Reshape((6, 3, 4)))
        # (6, 3, 4)

        self.encoder_layers.append(Convolution2D(4, 5, 5, activation='relu', border_mode='same'))
        # (6, 3, 4)

        self.encoder_layers.append(UpSampling2D((3, 3)))
        # (18, 9, 4)

        self.encoder_layers.append(Convolution2D(8, 5, 5, activation='relu', border_mode='same'))
        # (18, 9, 8)

        self.encoder_layers.append(UpSampling2D((3, 3)))
        # (54, 27, 8)

        self.encoder_layers.append(Convolution2D(16, 3, 3, activation='relu', border_mode='same'))
        # (54, 27, 16)

        self.encoder_layers.append(UpSampling2D((2, 2)))
        # (108,94,16)

        self.encoder_layers.append(
            Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', 
                          name=self.img_out))

        x = self.z
        for layer in self.encoder_layers:
            x = layer(x)
        self.outputs[self.img_out] = x

    def make_auxillary_outputs(self):
        x = Dense(2*self.latent_dim, activation='relu')(self.z)
        self.outputs[self.gasdet_out] = Dense(1, name=self.gasdet_out)(x)

        x = Dense(2*self.latent_dim, activation='relu')(self.z)
        self.outputs[self.label_out] = Dense(2, activation='softmax', name=self.label_out)(x)

    def define_models_from_outputs(self):
        self.vae = Model(input=[self.input_img],
                         output=[self.outputs[self.img_out],
                                 self.outputs[self.gasdet_out],
                                 self.outputs[self.label_out] ])
        self.encoder = Model(self.input_img, self.z_mean)

        self.decoder_input = Input(shape=(self.latent_dim,))
        x = self.decoder_input
        for layer in self.encoder_layers:
            x = layer(x)
        self.generator = Model(self.decoder_input, x)

    def make_losses(self):
        self.loss_fns[self.img_out] = self.img_decoded_mean_loss_fn
        self.loss_fns[self.gasdet_out] = self.gasdet_loss_fn
        self.loss_fns[self.label_out] = self.label_loss_fn
        
    def img_decoded_mean_loss_fn(self, x, y):
        x = Flatten()(x)
        y = Flatten()(y)
        self.loss_results[self.img_part_of_img_loss] = objectives.binary_crossentropy(x,y)
        self.loss_results[self.k1_part_of_img_loss] = -0.5 * kb.sum(1 + self.z_log_var - kb.square(self.z_mean) - kb.exp(self.z_log_var), axis=-1)
        self.loss_results[self.k1_part_of_img_loss] *= self.k1_loss_weight
        return self.loss_results[self.img_part_of_img_loss] + self.loss_results[self.k1_part_of_img_loss]

    def label_loss_fn(self, x, y):
        self.loss_results[self.label_out] = objectives.binary_crossentropy(x, y)
        return self.loss_results[self.label_out]

    def gasdet_loss_fn(self, x, y):
        self.loss_results[self.gasdet_out] = objectives.mean_squared_error(x, y)
        return self.loss_results[self.gasdet_out]

    def compile_vae(self, optimizer):
        self.vae.compile(optimizer=optimizer,
                         loss=self.loss_fns,
                         metrics=self.metrics,
                         loss_weights=self.loss_weights)
        
                         
