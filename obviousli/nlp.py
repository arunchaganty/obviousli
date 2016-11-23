#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP primitives.
"""

import logging
from copy import deepcopy
from collections import defaultdict
from itertools import chain

from stanza.nlp import AnnotatedSentence, AnnotatedDependencyParseTree, CoreNLP_pb2

logger = logging.getLogger(__name__)

# Operations to implement
# "find" -- returns tokens that match a dependency label.
# "drop subtree" -- linked list.
# "get subtree" -- return string of the subtree.

def get_tokens(graph):
    return sorted(set(chain(graph.roots, graph.graph.keys(), (i for vs in graph.graph.values() for i, _ in vs))))

# Dependency graphs are represented as an adjacency list.
def find_edges(dep_graph, label=None, fn=None):
    """
    Find all edges that match @fn. If @label is provided; match all edges with a label prefixed by @label.
    @fn(int, int, str) -> bool: matching function on the edge (source, target, label).
    @returns generator of all matching edges.
    """
    if fn is None and label is not None:
        fn = lambda _, __, l: l.startswith(label)

    # Creating a shallow copy of dep_graph because, as a default_dict, it's
    # possible for a new entry to be added, changing the keys but not
    # the values.
    graph = dict(dep_graph.graph)
    for i, edges in graph.items():
        for j, l in edges:
            if fn(i, j, l):
                yield (i, j, l)

def drop_subtree(dep_graph, token_idx):
    """
    Drop the node and subtree rooted at @token_idx.
    """
    graph = deepcopy(dep_graph.graph)
    roots = list(dep_graph.roots)
    if dep_graph.inv_graph[token_idx]:
        for parent, label in dep_graph.inv_graph[token_idx]:
            graph[parent].remove((token_idx,label))
    else:
        roots.remove(token_idx)
    for i in dep_graph.descendants(token_idx):
        if graph[i]: del graph[i]

    # TODO: drop punct

    new_graph = AnnotatedDependencyParseTree.from_graph(graph, roots)

    assert new_graph.tokens == get_tokens(new_graph)

    return to_sentence(new_graph, dep_graph.sentence)

def keep_subtree(dep_graph, token_idx):
    """
    Keep the subtree rooted @token_idx.
    """
    # Copy over subtree.
    graph = defaultdict(list)
    roots = [token_idx]
    for i in dep_graph.descendants(token_idx):
        graph[i] = list(dep_graph.graph[i])

    # If the root has a 'mark' node, drop it and all its children.
    for child, label in [(child, label) for child, label in graph[token_idx] if label == "mark"]:
        for child_ in dep_graph.descendants(child, [token_idx]):
            del graph[child_]
        graph[token_idx].remove((child, label))

    new_graph = AnnotatedDependencyParseTree.from_graph(graph, roots)

    assert new_graph.tokens == get_tokens(new_graph)


    return to_sentence(new_graph, dep_graph.sentence)

def to_sentence(dep_graph, sentence):
    """
    convert an AnnotatedDependencyTree into a AnnotatedSentence by keeping only the tokens that are a part of the new dep tree.
    """
    old_pb = sentence.pb
    new_pb = CoreNLP_pb2.Sentence()
    new_pb.characterOffsetBegin = old_pb.characterOffsetBegin
    new_pb.characterOffsetEnd = old_pb.characterOffsetBegin
    new_pb.sentenceIndex = old_pb.sentenceIndex
    new_pb.tokenOffsetBegin = old_pb.tokenOffsetBegin
    new_pb.tokenOffsetEnd = old_pb.tokenOffsetBegin
    # Drop tokens that aren't part of the dependency graph.

    token_map = {}
    for i, j in enumerate(dep_graph.tokens):
        t = new_pb.token.add()
        t.CopyFrom(old_pb.token[j])
        token_map[j] = i

    # Update pointers from graph.
    roots = [token_map[j] for j in dep_graph.roots]
    graph = defaultdict(list, {token_map[n]: [(token_map[j], lbl) for j, lbl in ns] for n, ns in dep_graph.graph.items()})

    dep_graph_ = AnnotatedDependencyParseTree.from_graph(graph, roots)
    new_pb.enhancedPlusPlusDependencies.CopyFrom(dep_graph_.pb)
    return AnnotatedSentence.from_pb(new_pb)
