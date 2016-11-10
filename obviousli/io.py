#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to read/write from various files.
"""

import gzip
import json
from collections import namedtuple

from tqdm import tqdm

from obviousli.defs import State

def load_states_from_file(f):
    """
    Load a stream of states from a file.
    """
    return (State.from_json(json.loads(line)) for line in tqdm(list(f)))

def process_snli_data(datafile):
    """
    Read the SNLI data and return output as ((sentence1, sentence2), label)
    """
    LABELS = [
        "neutral",
        "contradiction",
        "entailment",
        ]
    LABEL_MAP = {label:index for index, label in enumerate(LABELS)}


    Example = namedtuple('Example', ["sentence1", "sentence2", "tokens1", "tokens2", "label"])
    for line in datafile:
        obj = json.loads(line)
        tokens1 = obj["sentence1_binary_parse"].replace(r'(', '').replace(r')', '').split()
        tokens2 = obj["sentence2_binary_parse"].replace(r'(', '').replace(r')', '').split()
        label = obj["gold_label"]
        if label == "-": continue # Skip
        yield Example(obj["sentence1"], obj["sentence2"], tokens1, tokens2, LABEL_MAP[label])

def GzippableFileType(*args, **kwargs):
    """
    A wrapper around file type that transparently handles gzip files.
    """
    def ret(fname):
        if fname.endswith(".gz"):
            return gzip.open(fname, *args, **kwargs)
        else:
            return open(fname, *args, **kwargs)
    return ret
