#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from abc import abstractmethod, abstractclassmethod

from tqdm import tqdm
import ipdb

import numpy as np
from keras.models import model_from_json
from keras.models import Model as KerasModel
from keras.layers import Input, Embedding, AveragePooling1D, Flatten, Dense, Activation, Merge, TimeDistributed

from .util import grouper
from .embeddings import CrossUnigramEmbedder

class BaseModel(KerasModel):
    def __init__(self, *args, **kwargs):
        self.queue = []
        self.batch_size = kwargs.get('batch_size', 10)
        super(BaseModel, self).__init__(*args, **kwargs)

    @classmethod
    def load(cls, fname_prefix):
        """
        Load model and weights from a file.
        """
        fname_prefix = os.path.join(fname_prefix, cls.__name__)
        model_fname = fname_prefix + ".model"
        weights_fname = fname_prefix + ".weights"

        with open(model_fname, "r") as f:
            json = f.read()
        model = model_from_json(json, custom_objects={cls.__name__:cls})
        model.load_weights(weights_fname)

        return model

    def save(self, fname_prefix):
        """
        Save the model and weights in a file.
        """
        fname_prefix = os.path.join(fname_prefix, self.__class__.__name__)
        model_fname = fname_prefix + ".model"
        weights_fname = fname_prefix + ".weights"

        with open(model_fname, "w") as f:
            f.write(self.to_json())
        self.save_weights(weights_fname, overwrite=True)

    @abstractclassmethod
    def build(cls, **kwargs):
        """
        Build the model.
        @returns - Model.
        """
        pass

    def enqueue(self, example):
        """
        Appends @example to list of elements to be queued.
        @example - a tuple of (input, output)
        """
        self.queue.append(example)

    @abstractmethod
    def update(self):
        pass

    @abstractclassmethod
    def embedder(cls):
        """
        Return an embedding class that stores state representations.
        """
        pass

class ActorModel(BaseModel):
    """
    Type signature: state, action -> value
    """
    def __init__(self, *args, **kwargs):
        super(ActorModel, self).__init__(*args, **kwargs)

    def predict_on_state(self, state):
        return self.predict(state.representation)

    def predict_on_state_batch(self, batch):
        return self.predict_on_batch([state.representation for state in batch])

    def predict_on_state_action(self, state, action):
        return self.predict([state.representation, action.representation])

    def predict_on_state_action_batch(self, batch):
        return self.predict_on_batch([[state.representation for state, _ in batch], [action.representation for _, action in batch]])

class CriticModel(BaseModel):
    """
    Type signature: state -> value.
    """
    def __init__(self, *args, **kwargs):
        super(CriticModel, self).__init__(*args, **kwargs)

    def predict_on_state(self, state):
        return self.predict(state.representation)

    def predict_on_state_batch(self, batch):
        return self.predict_on_batch([state.representation for state in batch])

class EntailmentModel(BaseModel):
    """An entailment model"""
    output_shape = 3 # [neutral, entailment, contradiction]

    def __init__(self, *args, **kwargs):
        self.input_length = kwargs.pop('input_length')
        super(EntailmentModel, self).__init__(*args, **kwargs)

    def predict_on_state(self, state):
        return self.predict(state.representation)

    def predict_on_state_batch(self, batch):
        return self.predict_on_batch([state.representation for state in batch])

    def update(self):
        L = self.input_length
        L_ = self.output_shape
        logging.info("Updating %s...", self.__class__.__name__)
        for batch in tqdm(grouper(self.batch_size, self.queue), total=int(len(self.queue)/self.batch_size)):
            x = np.zeros((len(batch),L))
            y = np.zeros((len(batch),L_))
            for i, (state, label) in enumerate(batch):
                x[i, :] = state.representation
                y[i, label.value] = 1
            self.train_on_batch(x, y)
        self.queue = []

# ==== end boilerplate

class LexicalCrossUnigramModel(EntailmentModel):
    def __init__(self, *args, **kwargs):
        super(LexicalCrossUnigramModel, self).__init__(*args, **kwargs)

    @classmethod
    def embedder(cls):
        return CrossUnigramEmbedder

    @classmethod
    def build(cls, **kwargs):
        """
        Construct model to take as input a sparse vector of integers
        (corresponding to cross-unigrams).
        Maps to an output space of 3.
        """
        vocab_size = kwargs.pop('vocab_size')
        input_length = kwargs['input_length']
        output_type = 'softmax'
        output_shape = cls.output_shape

        x = Input(shape=(input_length,))
        z = Embedding(vocab_size+1, output_shape, input_length=input_length)(x)
        z = AveragePooling1D(pool_length=input_length)(z)
        z = Flatten()(z)
        z = Activation(output_type)(z)
        return cls(input=[x], output=[z], **kwargs)

class CrossUnigramModel(EntailmentModel):
    def __init__(self, *args, **kwargs):
        super(CrossUnigramModel, self).__init__(*args, **kwargs)

    @classmethod
    def embedder(cls):
        return CrossUnigramEmbedder

    @classmethod
    def build(cls, **kwargs):
        """
        Construct model to take as input a sparse vector of integers
        (corresponding to cross-unigrams).
        Maps to an output space of 3.
        """
        input_length = kwargs['input_length']
        emb_matrix = kwargs.pop('emb_matrix')
        vocab_size, emb_dim = emb_matrix.shape
        output_type = 'softmax'
        output_shape = cls.output_shape

        # These are two input sentences (source, target)
        x1 = Input(shape=(input_length,))
        x2 = Input(shape=(input_length,))

        # Embed them.
        E = Embedding(vocab_size, emb_dim, input_length=input_length, weights=emb_matrix, trainable=False)
        z1, z2 = E(x1), E(x2)

        # 1 dense layer
        P = TimeDistributed(Dense(emb_dim, activation='relu'))
        z1, z2 = P(z1), P(z2)

        # add the two layers together (silly)
        z = Merge([z1, z2], mode='concat')
        z = Flatten()(z)
        z = Dense(output_shape, activation=output_type)(z)
        return cls(input=[x1, x2], output=[z], **kwargs)
