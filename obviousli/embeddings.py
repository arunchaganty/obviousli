#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The new age feature factories.
"""

import logging
import pickle
from abc import abstractmethod, abstractclassmethod
from collections import Counter

class Embedder(object):
    @abstractclassmethod
    def construct(cls, **kwargs):
        """
        Constructs an embedding over the input data.
        Usually this involves constructing state map.
        """
        pass

    @abstractclassmethod
    def load(cls, fname):
        """
        Load embedder from a file.
        """
        pass

    @abstractmethod
    def save(self, fname):
        """Save to a file."""
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

    def __init__(self, word_map, **kwargs):
        super(CrossUnigramEmbedder, self).__init__(**kwargs)
        self.word_map = word_map
        self.preserve_case = kwargs.get('preserve_case', False)

    def save(self, fname):
        with open(fname, "w") as f:
            pickle.dump((self.preserve_case, self.word_map), f)

    @classmethod
    def load(cls, fname):
        with open(fname, "r") as f:
            preserve_case, word_map = pickle.load(f)
        return cls(word_map, preserve_case=preserve_case)

    @classmethod
    def construct(cls, **kwargs):
        """
        @data consists of a set of states.
        """
        data = kwargs.pop('data')
        preserve_case = kwargs.get('preserve_case', False) # only count pairs that occur at least 10 times.
        cutoff = kwargs.get('cutoff', 9) # only count pairs that occur at least 10 times.

        assert preserve_case == False # TODO(chaganty): fix code to save preserve_case state

        logging.info("Constructing %s...", cls.__name__)
        words = Counter()
        for state in data:
            words.update((tok_s.word.lower(), tok_t.word.lower()) for tok_s in state.source.tokens for tok_t in state.target.tokens)
        words = sorted(w for w, cnt in words.items() if cnt > cutoff)
        # keep the top
        word_map = {(word_s, word_t) : i+1 for i, (word_s, word_t) in enumerate(words)} # 0 reserved for null.
        logging.info("Completed with %d words", len(word_map))
        return cls(word_map, **kwargs)

    def embed(self, datum):
        """
        @datum: is a state
        @returns: a sequence of integers corresponding to all word pairs between the two sentences.
        """
        state = datum
        M = self.word_map
        return [M.get((t1.word.lower(),t2.word.lower()), 0) for t1 in state.source.tokens for t2 in state.target.tokens]

class DistributedEmbedder(Embedder):
    """
    A wrapper around a word vector model.
    """
    def __init__(self, indices, weights, **kwargs):
        self.indices = indices
        self.weights = weights
        self.count, self.dim = weights.shape
        self.preserve_case = kwargs.get('preserve_case', False)
        self.unknown_token = kwargs.get('unknown_token', 'unk')

    def save(self, fname):
        mmap_fname = fname + '.mmap'
        index_fname = fname + '.index'

        # TODO: copy the weights in current file to new file, if they aren't the same.
        assert self.weights.filename == os.path.abspath(mmap_fname), "Saved mmap file {} different from provided path {}".format(self.weights.filename, os.path.abspath(mmap_fname))
        with open(index_fname, 'w') as f:
            json.dump({
                "indices": self.indices,
                "count" : self.count
                "dim": self.dim,
                "preserve_case":self.preserve_case,
                "unknown_token":self.unknown_token,
                }, f)

    @classmethod
    def load(cls, fname):
        mmap_fname = fname + '.mmap'
        index_fname = fname + '.index'

        assert os.path.exists(index_fname) and os.path.exists(mmap_fname), "Couldn't find path to files."

        with open(index_fname) as f:
            obj = json.load(f)
        weights = np.memmap(mmap_fname, mode='r', shape=(len(obj["indices"]), obj["dim"]), dtype=np.float32)
        return cls(obj['indices'], weights, obj['dim'], obj['preserve_case'], obj['unknown_token'])

    @classmethod
    def construct(cls, **kwargs):
        """
        Construct a word vector map from a file
        """
        ifname = kwargs.get('ifname')
        preserve_case = kwargs.get('preserve_case', False)
        unknown_token = kwargs.get('unknown', 'unk')

        ofname = kwargs.get('ofname', '')
        mmap_fname = ofname + '.mmap'
        index_fname = ofname + '.index'

        logging.info("Reading word vectors")
        with open(ifname, 'r') as f:
            line = next(f)
            parts = line.split()
            dim = len(parts) - 1 # -1 for the token
            n_lines = 1
            for _ in f: n_lines += 1
        logging.info("Found %d %d-dim vectors", n_lines, dim)

        # Allocate a vector.
        indices = {}
        weights = np.memmap(mmap_fname, mode='w+', shape=(n_lines, dim), dtype=np.float32)

        # get the number of vectors
        with open(ifname, 'r') as f:
            for i, line in enumerate(f):
                parts = line.split()
                tok = parts[0]
                if preserve_case:
                    tok = tok.lower()
                vec = array([float(x) for x in parts[1:]])
                assert len(vec) == dim, "Incorrectly sized vector"

                indices[tok] = i
                weights[i, :] = vec
            # TODO(chaganty): create a new random vector for unknown tokens. 
            assert unknown_token in indices, "Unknown token not defined in word vectors"
            assert len(indices) == n_lines, "Had more than one line in file map to the same token"
        return cls(indices, weights, **kwargs)

    def __getitem__(self, tok):
        if self.preserve_case:
            tok = tok.lower()
        if tok not in self.indices:
            tok = self.unknown_token
        return self.indices[tok]

    def __call__(self, tok):
        return self.weights[self[tok], :]

    def embed(self, datum):
        """
        @datum: is a sentence (with a tokens method)
        @returns: a sequence of integers corresponding to all tokens in the sentence.
        """
        return self.project_sentence(datum.source), self.project_sentence(datum.target) 

    def project_sentence(self, sentence):
        """
        Return a list of indices into the word vectors
        """
        return [self[t] for t in sentence.tokens]

    @classmethod
    def embed_sentence(self, sentence):
        """
        Return the list of tokens embedded as a matrix.
        """
        return [self(t) for t in sentence.tokens]

