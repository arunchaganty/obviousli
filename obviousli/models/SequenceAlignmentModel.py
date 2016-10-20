#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence model with word-to-word attention.
As described in Rocktäschel, T., Grefenstette, E., Moritz, K., Deepmind, H. G., Kočisk, T., & Blunsom, P. (n.d.). REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION.
"""

from . import EntailmentModel
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Merge, Input, AveragePooling1D, merge, Embedding
from util import WordEmbeddings

class SequenceAlignmentModel(EntailmentModel):
    """
    The basic model encodes both sentences using average pooling and
    trains a single layer model on the sentence encodings.
    """
    @classmethod
    def build(cls, **kwargs):
        """
        Combine the sentence embeddings x1, x2 to produce an entailment.
        """
        x1 = Input()
        x2 = Input()

        # Concatenate x1 :: x2
        # Do a forward LSTM over x1, initializing state for LSTM over x2.
        # Predict on last hidden state, h_N.

        # Adding attention
        # Take the output of the premise LSTM, Y and combine with the last hidden state (after hypothesis) h_N.
        # M = tanh(W_y Y + W_h h_N × 1)
        # α is softmax(w' M)
        # r = Y α^T
        # Predict with h^* = tanh(W_p r + W_x h_N)

        # Adding word-word attention
        # M_t = tanh(W_y Y + (W_h h_t + W_r r_{t-1}) × 1)
        # α_t is softmax(w' M_t)
        # r_t = Y α_t^T + tanh(W_t r_{t-1})
        # h^* = tanh(W_p r_N + W_x h_N)


        y = None

        return EntailmentModel(input=[x1,x2], output=[y])


