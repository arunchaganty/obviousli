#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The new age feature factories.
"""

import logging
from abc import abstractmethod, abstractclassmethod

class Embedder(object):
    @abstractclassmethod
    def construct(cls, data):
        """
        Constructs an embedding over the input data.
        Usually this involves constructing state map.
        """
        pass

    @abstractmethod
    def embed(self, datum):
        """Embed datum into appropriate embedding space."""
        pass

class CrossUnigramEmbedder(Embedder):
    """
    Constructs an embedding wherein each token of the state.source and
    state.target are embedded as an integer.
    """

    def __init__(self, word_map):
        self.word_map = word_map

    @classmethod
    def construct(cls, data):
        """
        @data consists of a set of states.
        """
        logging.info("Constructing %s...", cls.__name__)
        words = set([])
        for state in data:
            words.update((tok_s.word.lower(), tok_t.word.lower()) for tok_s in state.source.tokens for tok_t in state.target.tokens)
        word_map = {(word_s, word_t) : i for i, (word_s, word_t) in enumerate(sorted(words))}
        logging.info("Completed with %d words", len(word_map))
        return cls(word_map)

    def embed(self, datum):
        """
        @datum: is a state
        @returns: a sequence of integers corresponding to all word pairs between the two sentences.
        """
        state = datum
        M = self.word_map
        return [M[(t1.word.lower(),t2.word.lower())] for t1 in state.source.tokens for t2 in state.target.tokens]
