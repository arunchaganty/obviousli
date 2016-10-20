#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility routines
"""

import os
import json
import itertools
import logging
from collections import namedtuple, defaultdict, Counter
import numpy as np
from numpy import array

def edit_distance(l1, l2):
    """
    Computes edit distance between two sequences, l1, l2
    """
    if len(l1) < len(l2):
        return edit_distance(l2, l1)

    # len(l1) >= len(l2)
    if len(l2) == 0:
        return len(l1)

    previous_row = list(range(len(l2) + 1))
    for i, c1 in enumerate(l1):
        current_row = [i + 1]
        for j, c2 in enumerate(l2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than l2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def normalized_edit_distance(l1, l2):
    """
    @returns: `edit_distance` normalized by length of input (lies between 0, 1).
    """
    return edit_distance(l1, l2) / max(len(l1), len(l2))

def process_snli_data(datafile):
    """
    Read the SNLI data and return output as ((sentence1, sentence2), label)
    """
    Example = namedtuple('Example', ["sentence1", "sentence2", "tokens1", "tokens2", "label"])
    for line in datafile:
        obj = json.loads(line)
        tokens1 = obj["sentence1_binary_parse"].replace(r'(', '').replace(r')', '').split()
        tokens2 = obj["sentence2_binary_parse"].replace(r'(', '').replace(r')', '').split()
        label = obj["gold_label"]
        if label == "-": continue # Skip
        yield Example(obj["sentence1"], obj["sentence2"], tokens1, tokens2, LABEL_MAP[label])

def pad_zeros(arr, length):
    """
    Pad zeros to make all input fixed length
    """
    return arr[:length] + [0] * max(0, length - len(arr))

LABELS = [
    "neutral",
    "contradiction",
    "entailment",
    ]
LABEL_MAP = {label:index for index, label in enumerate(LABELS)}

def __vectorize_data(obj, max_length):
    """
    Returns a vectorized representation of the input data.
    Each sentence is converted into a sequence of token indices.
    Each output vector is translated as a one hot vector
    """
    x1 = pad_zeros(WordEmbeddings.project_sentence(obj.sentence1), max_length)
    x2 = pad_zeros(WordEmbeddings.project_sentence(obj.sentence2), max_length)
    y = [0, 0, 0]
    if obj.label is not None:
        y[obj.label] = 1

    return array(x1), array(x2), array(y)

def vectorize_data(objs, max_length):
    """
    Returns a vectorized representation of the input data.
    Each sentence is converted into a sequence of token indices.
    Each output vector is translated as a one hot vector
    """
    if isinstance(objs, list):
        X1, X2, Y = [], [], []
        for obj in objs:
            x1, x2, y = __vectorize_data(obj, max_length)
            X1.append(x1)
            X2.append(x2)
            Y.append(y)
        return array(X1), array(X2), array(Y)
    else:
        return __vectorize_data(objs, max_length)

class Scorer(object):
    """
    This object keeps running track of scores while training.
    """
    def __init__(self, metrics):
        self.metrics = metrics
        self.score = [0. for _ in self.metrics]
        self.n = 0

    def update(self, score, n_items):
        """
        Update metrics
        """
        self.n += n_items
        for i, val in enumerate(score):
            self.score[i] += n_items * (val - self.score[i])/self.n

    def __str__(self):
        return "\t".join(str(name) + " " + str(score) for name, score in zip(self.metrics, self.score))

    def keys(self):
        """
        Return metric types
        """
        return self.metrics

    def values(self):
        """
        Return values of the matrics as a string.
        """
        return self.score

def test_scorer():
    scorer = Scorer(["x","y"])
    items = [(-1,-1), (1,1), (1,1)]
    avgs =  [(-1,-1), (0.,0.), (1./3.,1./3.)]
    for item, avg in zip(items, avgs):
        scorer.update(item, 1)
        print(scorer.values(), avg)
        assert np.allclose(scorer.values(), avg)

def grouper(n, iterable):
    """
    grouper(3, 'ABCDEFG') --> 'ABC', 'DEF', 'G'
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class WordEmbeddings(object):
    """
    A wrapper around a word vector model.
    """
    class __Singleton(dict): 
        """
        Singleton class for word embeddings.
        """
        def __init__(self, index_map, weights, dim, preserve_case=False, unknown='unk'):
            dict.__init__(self, index_map)
            self.weights = weights
            self.dim = dim
            self.preserve_case = preserve_case
            self.unknown = unknown

        def __getitem__(self, key):
            if not self.preserve_case:
                key = str.lower(key)
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return dict.__getitem__(self, self.unknown)

        def __setitem__(self, key, val):
            if not self.preserve_case:
                key = str.lower(key)
            return dict.__setitem__(self, key, val)

    instance = None
    def __init__(self):
        if WordEmbeddings.instance is None:
            raise AttributeError("Word embeddings must first be initalized using from_file")

    def __getattr__(self, name):
        """Dispatch all attribute calls to singleton instance"""
        return getattr(self.instance, name)

    def __len__(self):
        return len(self.instance)

    @classmethod
    def from_file(cls, f, preserve_case=False, unknown="unk", mmap_fname=".mmap", index_fname='.index'):
        """
        Construct a word vector map from a file
        """
        logging.info("Reading word vectors")
        if os.path.exists(index_fname) and os.path.exists(mmap_fname):
            with open(index_fname) as f:
                obj = json.load(f)
            logging.info("Using cached version on disk.")
            weights = np.memmap(mmap_fname, mode='r', shape=(len(obj["indices"]), obj["dim"]), dtype=np.float32)

            cls.instance = WordEmbeddings.__Singleton(obj["indices"], weights, obj["dim"], obj["preserve_case"], obj["unknown"])
        else:
            mapping = {}
            dim = None
            for line in f:
                parts = line.split()
                tok = parts[0]
                vec = array([float(x) for x in parts[1:]])
                if dim is None:
                    dim = len(vec)
                assert dim == len(vec), "Incorrectly sized vector"
                mapping[tok] = vec
            assert unknown in mapping, "Unknown token not defined in word vectors"

            # Create an index map and compress dictionary into a matrix.
            indices = {}
            weights = np.memmap(mmap_fname, mode='w+', shape=(len(mapping), dim), dtype=np.float32)
            for i, (key, vec) in enumerate(mapping.items()):
                indices[key] = i
                weights[i,:] = vec

            with open(index_fname, 'w') as f:
                json.dump({
                    "indices": indices,
                    "dim":dim,
                    "preserve_case":preserve_case,
                    "unknown":unknown,
                    }, f)

            cls.instance = WordEmbeddings.__Singleton(indices, weights, dim, preserve_case, unknown)
            logging.info("Done. Loaded %d vectors.", len(cls.instance.weights))

    @classmethod
    def from_filename(cls, fname, preserve_case=False, unknown="unk"):
        """
        Construct a word vector map from a file
        """
        with open(fname) as f:
            mmap_fname = fname+'.mmap'
            WordEmbeddings.from_file(f, preserve_case, unknown, mmap_fname)

    @classmethod
    def project_sentence(cls, toks, max_length=40):
        """
        Return a list of indices into the word vectors
        """
        return [cls.instance[t] for t in toks[:max_length]]

    @classmethod
    def embed_sentence(cls, indices, max_length=40):
        """
        Return the list of tokens embedded as a matrix.
        """
        return cls.instance.weights[indices[:max_length],:]

    @classmethod
    def embed_sentences(cls, sentences, max_length=40):
        """
        Return the list of tokens embedded as a matrix.
        """
        return array([[cls.embed_sentence(toks, max_length)] for toks in sentences])

class SparseWordEmbeddings(object):
    """
    A wrapper around a word vector model.
    """
    class __Singleton(dict): 
        """
        Singleton class for word embeddings.
        """
        def __init__(self, index_map, dim, preserve_case=False, unknown='unk'):
            dict.__init__(self, index_map)
            self.dim = dim
            self.preserve_case = preserve_case
            self.unknown = unknown

        def __getitem__(self, key):
            if not self.preserve_case:
                key = str.lower(key)
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return dict.__getitem__(self, self.unknown)

        def __setitem__(self, key, val):
            if not self.preserve_case:
                key = str.lower(key)
            return dict.__setitem__(self, key, val)

    instance = None
    def __init__(self):
        if WordEmbeddings.instance is None:
            raise AttributeError("Word embeddings must first be initalized using from_data")

    def __getattr__(self, name):
        """Dispatch all attribute calls to singleton instance"""
        return getattr(self.instance, name)

    def __len__(self):
        return len(self.instance)

    @classmethod
    def from_data(cls, words, preserve_case=False, unknown="unk", index_fname='.index'):
        """
        Construct a word vector map from a file
        """
        logging.info("Saving words")
        if os.path.exists(index_fname):
            with open(index_fname) as f:
                obj = json.load(f)
            logging.info("Using cached version on disk.")
            cls.instance = cls.__Singleton(obj["indices"], obj["dim"], obj["preserve_case"], obj["unknown"])
        else:
            if not preserve_case:
                words = map(str.lower, words) 
            indices = {word : i for i, word in enumerate(sorted(set(words)))}
            dim = 1
            preserve_case = True
            with open(index_fname, 'w') as f:
                json.dump({
                    "indices": indices,
                    "dim":dim,
                    "preserve_case":preserve_case,
                    "unknown":unknown,
                    }, f)
            cls.instance = cls.__Singleton(indices, dim, preserve_case, unknown)
            logging.info("Done. Loaded %d vectors.", len(cls.instance))

    @classmethod
    def from_filename(cls, fname, preserve_case=False, unknown="unk"):
        """
        Construct a word vector map from a file
        """
        with open(fname) as f:
            mmap_fname = fname+'.mmap'
            WordEmbeddings.from_file(f, preserve_case, unknown, mmap_fname)

    @classmethod
    def project_sentence(cls, toks, max_length=40):
        """
        Return a list of indices into the word vectors
        """
        return [cls.instance[t] for t in toks[:max_length]]

    @classmethod
    def embed_sentence(cls, indices, max_length=40):
        """
        Return the list of tokens embedded as a matrix.
        """
        return cls.instance.weights[indices[:max_length],:]

    @classmethod
    def embed_sentences(cls, sentences, max_length=40):
        """
        Return the list of tokens embedded as a matrix.
        """
        return array([[cls.embed_sentence(toks, max_length)] for toks in sentences])

def to_table(data, row_labels, column_labels):
    """Pretty print tables"""
    # Convert data to strings
    data = [["%.2f"%v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret

class ConfusionMatrix(object):
    """
    Keeping track of the confusion matrix.
    """

    def __init__(self, labels):
        self.labels = labels
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def print_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        print(to_table(data, self.labels, ["go\\gu"] + self.labels))

    def summary(self):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))
        # Macro and micro average.


        print(to_table(data, self.labels + ["micro","macro"], ["label", "acc", "prec", "rec", "f1"]))

