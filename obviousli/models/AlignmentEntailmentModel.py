#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A pairwise attention model:
As described in Parikh, A. P., Täckström, O., Das, D., & Uszkoreit, J.
(2016). A Decomposable Attention Model for Natural Language Inference.
Retrieved from http://arxiv.org/abs/1606.01933
"""

from . import EntailmentModel
from layers import InnerAttention
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling1D, merge, TimeDistributed, LSTM

class AlignmentEntailmentModel(EntailmentModel):
    """
    The basic model encodes both sentences using average pooling and
    trains a single layer model on the sentence encodings.
    """
    @classmethod
    def build(cls, **kwargs):
        """
        Combine the sentence embeddings x1, x2 to produce an entailment.
        """
        input_shape = kwargs['input_shape']
        output_shape = kwargs.get('output_shape', cls.output_shape)
        emb_dim = kwargs.get('emb_dim', 200)
        g_dim = kwargs.get('g_dim', 200)
        h_dim = kwargs.get('h_dim', 200)
        p = kwargs.get('dropout_rate', 0.2)
        input_length, input_dim = input_shape

        # Inputs are a, b, following notation
        # Each of these is already encoded as a sentence.
        a = Input(input_shape)
        b = Input(input_shape)

        # Feedforward encoding of each (word) of a, b
        #P1, P2 = Dense(emb_dim, activation='relu'), Dense(emb_dim, activation='relu')
        P = TimeDistributed(Sequential([Dense(emb_dim, activation='relu', input_dim=input_dim), Dense(emb_dim, activation='relu')]))
        #P = LSTM(emb_dim, return_sequences=True)
        a_, b_ = P(a), P(b)

        # NOTE: For "intra-sentence" - first do attention over tokens in a_i and return concatenation of [a_i, a'_i]

        # 1. Attend
        # Attention is e_{ij} = a_i^T b_j
        Q = InnerAttention()
        α = Q([a_, b_]) # aligned with a
        β = Q([b_, a_]) # aligned with b

        # 2. Compare aligned phrases
        #G1, G2 = Dropout(p), TimeDistributed(Dense(g_dim, activation='relu'))
        G = lambda x: G2(G1(x))
        #G = TimeDistributed(Sequential([Dense(g_dim, activation
        G = TimeDistributed(Sequential([Dense(g_dim, activation='relu', input_dim=2*emb_dim), Dense(g_dim, activation='relu')]))
        # v_{1,i} = G([a_i, β_i])
        v1 = G(merge([a_, β], mode='concat'))
        # v_{2,i} = G([α_j, b_j])
        v2 = G(merge([b_, α], mode='concat'))

        # 3. Aggregate
        # SumPool over v_1 and v_2
        v1 = Flatten()(AveragePooling1D(input_length)(v1))
        v2 = Flatten()(AveragePooling1D(input_length)(v2))

        #H1, H2 = Dropout(p), Dense(h_dim, activation='relu')
        #H = lambda x: H2(H1(x))
        H = Sequential([Dense(h_dim, activation='relu', input_dim=2*g_dim), Dense(h_dim, activation='relu'),])
        # Predict using H[v_1, v_2]
        z = H(merge([v1, v2], mode='concat'))
        z = Dropout(p)(z)
        y = Dense(output_shape, activation='softmax')(z)

        return EntailmentModel(input=[a,b], output=[y])


