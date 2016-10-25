#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A lexical model assumes as input sparse tokens from a map.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.layers import Input, Dense

from . import EntailmentModel

class LexicalModel(EntailmentModel):
    def __init__(self, word_map, **kwargs):
        super(LexicalModel, self).__init__(**kwargs)
        self.word_map = word_map

    @classmethod
    def build(cls, word_map, **kwargs):
        """
        Construct model to take as input a sparse vector of integers
        (assume words have been mapped already).
        Maps to an output space of 3.
        """
        input_shape = (len(word_map)*len(word_map),)
        output_type = 'softmax'
        output_shape = cls.output_shape

        x = Input(shape=input_shape)
        z = Dense(output_shape, activation=output_type)(x)
        return LexicalModel(word_map, input=[x], output=[z])

    def encode_state(self, state):
        M, L = self.word_map, len(self.word_map)
        return [M[t1.word]*L + M[t2.word] for t1 in state.source.tokens for t2 in state.target.tokens]

    def predict(self, state):
        L = len(self.word_map) * len(self.word_map)
        x = np.zeros((1,L))
        x[:,self.encode_state(state)] = 1

        # Need to project tokens into vectors.
        return super(LexicalModel, self).predict(x)

    def update(self):
        L = len(self.word_map) * len(self.word_map) 
        L_ = self.output_shape
        x = np.zeros((len(self.queue),L))
        y = np.zeros((len(self.queue),L_))

        for i, (state, label) in enumerate(self.queue):
            x[i, self.encode_state(state)] = 1
            y[i, label.value] = 1
        super(LexicalModel, self).train_on_batch(x, y)
        self.queue = []

# - have a "train" subcommand that trains a model; given input / output.
# - Instantiate the model, pass through the training data and enqueue it all.
# - Call update once.
# - Save model.

