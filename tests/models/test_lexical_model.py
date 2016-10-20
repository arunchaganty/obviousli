#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pytest
from obviousli.defs import State, Truth
from obviousli.models.LexicalModel import LexicalModel
from keras import backend as K

@pytest.fixture
def word_map():
    words = "we eat bananas for lunch do n't .".split()
    return {word : i for i, word in enumerate(sorted(set(words)))}

@pytest.fixture
def backend():
    """
    A wrapper around the keras backend that properly deletes the
    tensorflow session after running.
    """
    if K.backend() == "tensorflow":
        import tensorflow as tf
        sess = tf.Session()
        K.set_session(sess)
        yield None
        del sess

class TestLexicalModel(object):
    def test_lexical_model_build(self, backend, word_map):
        model = LexicalModel.build(word_map)
        assert model is not None
        model.compile(
            optimizer='adagrad',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def test_lexical_model_predict(self, backend, word_map):
        model = LexicalModel.build(word_map)
        model.compile(
            optimizer='adagrad',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        value = model.predict(State.new("we eat bananas for lunch .", "we eat bananas for lunch ."))
        assert value.shape == (1,3)

    def test_lexical_model_update(self, backend, word_map):
        model = LexicalModel.build(word_map)
        model.compile(
            optimizer='adagrad',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        for i in range(10):
            model.enqueue((State.new("we eat bananas for lunch .", "we eat bananas for lunch ."), Truth.TRUE))
            model.enqueue((State.new("we eat bananas for lunch .", "we don't eat bananas for lunch ."), Truth.FALSE))
            model.enqueue((State.new("we eat bananas for lunch .", "we eat bananas ."), Truth.TRUE))
            model.enqueue((State.new("we eat bananas for lunch .", "we eat lunch ."), Truth.TRUE))

            model.update()

        value = model.predict(State.new("we eat bananas for lunch .", "we eat bananas for lunch .", Truth.TRUE))
        assert value.shape == (1,3)
        assert value.argmax() == Truth.TRUE.value
