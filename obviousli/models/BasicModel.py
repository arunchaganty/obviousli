#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A basic entailment model
"""
from . import SentenceEntailmentModel
from keras.layers import Dense, Dropout, merge

class BasicModel(SentenceEntailmentModel):
    """
    The basic model encodes both sentences using average pooling and
    trains a single layer model on the sentence encodings.
    """
    def __init__(self, **kwargs):
        super(BasicModel, self).__init__(**kwargs)

    @classmethod
    def combine_sentences(cls, x1, x2, **kwargs):
        """
        Combine the sentence embeddings x1, x2 to produce an entailment.
        """
        hidden_dimension = kwargs.get('hidden_dimension', 128)
        output_shape = kwargs.get('output_shape', cls.output_shape)
        output_type = kwargs.get('output_type', 'softmax')

        # Softmax on top of these.
        z = merge([x1,x2], mode="concat")
        #z = Flatten()(z)
        z = Dropout(0.5)(z)
        z = Dense(hidden_dimension, activation='relu')(z)
        z = Dropout(0.5)(z)
        y = Dense(output_shape, activation=output_type)(z)

        return y

