#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A basic entailment model
"""
from . import SentenceModel
from keras.layers import Dense, Dropout, Flatten, Input, merge, Reshape, Convolution2D, MaxPooling2D

class NGramSentenceModel(SentenceModel):
    """
    Encodes a sentence mdoel that does a max pooling over various
    convolutions and merges them into a single representation.
    """
    @classmethod
    def build(cls, **kwargs):
        input_shape = kwargs['input_shape']
        output_shape = kwargs.get('output_shape', 128)
        filter_size = kwargs.get('filter_size', 200)
        ngrams_windows = kwargs.get('ngram-windows', [2,])

        input_length, input_dim = input_shape
        x = Input(shape=input_shape)
        x_ = Reshape((1,input_length, input_dim))(x) # There is only 1 channel

        # Simulate n-gram windows using convolution and pooling
        window_models = {}
        for window_length in ngrams_windows:
            # Convolution filter is (window_length x input_dim) long.
            y = Convolution2D(filter_size, window_length, input_dim, activation='relu')(x_)
            # y should now be filter_size x input_length - window_length + 1 x 1
            y = MaxPooling2D(pool_size=(input_length - window_length + 1, 1))(y)
            # y should now be input_dim x filter_size
            window_models[window_length] = y

        # Merge these windows
        if len(window_models) > 1:
            z = merge(list(window_models.values()), mode='concat')
        else:
            z = window_models[ngrams_windows[0]]
        z = Dropout(0.5)(z)
        z = Flatten()(z)
        # Finally add another non-linearity for this layer.
        z = Dense(output_shape, activation='relu')(z)

        return SentenceModel(input=[x], output=[z])

