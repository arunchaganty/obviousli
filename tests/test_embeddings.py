#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import json

import pytest

from obviousli.defs import State
from obviousli.embeddings import LexicalBaselineEmbedder

@pytest.fixture
def test_data():
    """
    Load some data for the embedder.
    """
    with open("tests/data/snli_100.jsonl") as f:
        return [State.from_json(json.loads(line)) for line in f]

def test_lexical_baseline_embedder_construct(test_data):
    """
    Construct an embedder and test the presence of key words when
    threshold is present and not.
    """
    embedder = LexicalBaselineEmbedder.construct(data=test_data, threshold=0)
    assert "0-bleu" in embedder.map
    assert embedder.features[0] == "0-bleu"
    assert "1-length_difference" in embedder.map
    assert "2-word_overlap_total_abs" in embedder.map
    assert "2-word_overlap_total_rel" in embedder.map
    assert "2-word_overlap_pos_abs" in embedder.map
    assert "2-word_overlap_pos_rel" in embedder.map
    assert "3-hypothesis:the" in embedder.map
    assert "3-hypothesis:child" in embedder.map
    assert "4-cross:NN:grass++child" in embedder.map
    assert len(embedder.map) == 2645

def test_lexical_baseline_embedder_embed(test_data):
    """
    Test that an embedder is able to embed sentences.
    """
    embedder = LexicalBaselineEmbedder.construct(data=test_data, threshold=0)
    datum = test_data[0]
    feats = dict(embedder.embed(datum))
    for idx in feats: assert idx < len(embedder.map)
    assert embedder.map['0-bleu'] in feats and feats[embedder.map['0-bleu']] > 0
    assert embedder.map['1-length_difference'] in feats and feats[embedder.map['1-length_difference']] == 5
    assert embedder.map['3-hypothesis:hugging'] in feats
    assert embedder.map['4-cross:NNS:women++sisters'] in feats

def test_lexical_baseline_embedder_save_load(tmpdir, test_data):
    """
    Test save/load feature of embedder.
    """
    embedder = LexicalBaselineEmbedder.construct(data=test_data, threshold=0)
    fname = tmpdir.join("tmp.txt")
    embedder.save(str(fname))
    assert fname.exists()
    embedder_ = embedder.load(str(fname))

    assert len(embedder.features) == len(embedder_.features)
    assert embedder.features == embedder_.features
