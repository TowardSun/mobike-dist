# -*- coding: utf-8 -*-

from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, AveragePooling2D, Concatenate
from keras.layers import GlobalAveragePooling2D, Dense
from keras.regularizers import l2
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import he_normal


class DenseConvModel:

    def __init__(self, input_shape, first_filters, nb_dense_block_layers,
                 growth_rate, compression, dropout=0.2, weight_decay=1e-4, lr=0.001):
        self.input_shape = input_shape
        self.first_filters = first_filters
        self.nb_filters = self.first_filters
        self.nb_dense_block_layers = nb_dense_block_layers
        self.growth_rate = growth_rate
        self.compression = compression
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr

    @staticmethod
    def conv_block(x, filters, kernel_size, padding='same', strides=1, activation=True, bn=True,
                   weight_decay=1e-4, dropout_rate=0.0):
        if bn:
            x = BatchNormalization(axis=-1,
                                   gamma_regularizer=l2(weight_decay))(x)
        if activation:
            x = Activation('relu')(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)

        x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer='he_normal',
                   padding=padding,
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        return x

    def transition_block(self, x, pooling=True):
        # compress channels
        self.nb_filters = int(self.nb_filters * self.compression)
        # BN-RELU-Conv(1*1), compression channels
        x = self.conv_block(x, self.nb_filters, kernel_size=1, padding='same', strides=1,
                            weight_decay=self.weight_decay, dropout_rate=self.dropout)

        if pooling:
            x = AveragePooling2D(3, strides=2)(x)
        return x

    def dense_conv_block(self, x, nb_layers):
        list_feat = [x]

        for i in range(nb_layers):
            # BN-RELU-Conv(1*1)
            x = self.conv_block(x, filters=self.growth_rate * 2, kernel_size=1, padding='valid', strides=1,
                                dropout_rate=self.dropout, weight_decay=self.weight_decay)
            # BN_RELU-Conv(1*3)
            x = self.conv_block(x, filters=self.growth_rate, kernel_size=3, padding='same',
                                dropout_rate=self.dropout, weight_decay=self.weight_decay)

            # dense connect
            list_feat.append(x)
            x = Concatenate(axis=-1)(list_feat)
            self.nb_filters += self.growth_rate

        return x

    def build_model(self):
        """
        MLP model for feature extraction
        :return:
        """
        inputs = Input(shape=self.input_shape)
        x = self.conv_block(inputs, self.first_filters, kernel_size=3, padding='same', bn=False, activation=False)

        for i in range(len(self.nb_dense_block_layers) - 1):
            x = self.dense_conv_block(x, self.nb_dense_block_layers[i])
            # add transition, scale down the time steps
            if i == 0:
                x = self.transition_block(x, pooling=True)
            else:
                x = self.transition_block(x, pooling=False)
        x = self.dense_conv_block(x, self.nb_dense_block_layers[-1])
        x = BatchNormalization(axis=-1, gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D(name='output')(x)

        outputs = Dense(1, kernel_initializer=he_normal(seed=0))(x)

        nn_model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(lr=self.lr)
        nn_model.compile(optimizer=optimizer, loss='mse')

        return nn_model
