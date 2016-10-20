# -*- coding: utf-8 -*-
"""
A basic sentence model
"""
from . import SentenceModel
from keras.layers import AveragePooling1D, Input, Flatten

class BasicSentenceModel(SentenceModel):
    """
    A basic sentence model that computes the sentence embedding using
    just average pooling.
    """
    @classmethod
    def build(cls, **kwargs):
        input_shape = kwargs.get('input_shape')
        input_length, _ = input_shape
        x = Input(shape=input_shape)
        z = AveragePooling1D(pool_length=input_length-1)(x)
        z = Flatten()(z)
        return SentenceModel(input=[x], output=[z])

