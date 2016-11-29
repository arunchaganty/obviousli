#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP primitives.
"""

import logging
from copy import deepcopy
from collections import defaultdict
from itertools import chain
import heapq

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

    # TODO(chaganty): drop punct as appropriate.
    # TODO(chaganty): handle appos.
    # TODO(chaganty): handle conj.
    new_graph = AnnotatedDependencyParseTree.from_graph(graph, roots)

    assert new_graph.tokens == get_tokens(new_graph)

    return _to_sentence(new_graph, dep_graph.sentence)

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
            if graph[child_]: del graph[child_]
        graph[token_idx].remove((child, label))

    new_graph = AnnotatedDependencyParseTree.from_graph(graph, roots)

    assert new_graph.tokens == get_tokens(new_graph)


    return _to_sentence(new_graph, dep_graph.sentence)

def _to_sentence(dep_graph, sentence):
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

def _score_mod(source, target):
    """
    Score based on Jaccard score of lemma overlap
    """
    s = set(source.lemmas)
    t = set(target.lemmas)

    return len(s.intersection(t))/len(s.union(t))

def _filter_mod(source, target, original_source):
    """
    Make sure that s, t pair have as many NNs, JJs and CDs in common as original source.
    """

    def keep_token(t):
        return t.pos.startswith("NN") or t.pos.startswith("JJ") or t.pos.startswith("CD")

    nnps_original = set([t.lemma for t in original_source if keep_token(t)])
    nnps_source = set([t.lemma for t in source if keep_token(t)])
    nnps_target = set([t.lemma for t in target if keep_token(t)])

    return nnps_original.intersection(nnps_target) == nnps_source.intersection(nnps_target)

def _mod_gen(sent):
    dp = sent.depparse()
    for label in ["ccomp", "nmod", "amod", "advmod", "ccomp", "acl", "appos", "advcl"]:
        for (_, j, _) in find_edges(dp, label=label):
            yield drop_subtree(dp, j)
            yield keep_subtree(dp, j)

def simplify(state_):
    """ Deterministic modification -- recursively simply until you get a
        sentence with all the named entities between the two entences."""

    candidates = [(-_score_mod(state_.source, state_.target), state_)]
    visited = set([str(state_.source)])

    best_cost, best_state = 1e10, None
    while len(candidates) > 0:
        cost, state = heapq.heappop(candidates)
        logger.debug("modifying %.2f %s", cost, state)

        # update best tracker
        if cost < best_cost:
            best_cost, best_state = cost, state

        # simplify the state.source
        for mod in _mod_gen(state.source):
            if _filter_mod(mod, state.target, state_.source):
                cost, state = -_score_mod(mod, state.target), state.replace(source=mod)
                if str(state.source) in visited:
                    logger.debug("already seen %.2f %s", cost, mod)
                else:
                    logger.debug("considering %.2f %s", cost, mod)
                    heapq.heappush(candidates, (cost, state))
                    visited.add(str(state.source))
            else:
                logger.debug("rejected %s", mod)
        # Trim candidates to top k.
        logger.debug("beam contains %d elements", len(candidates))
        candidates = candidates[:5]
    logger.info("best pair %.2f %d -> %.2f %d %s", -_score_mod(state_.source, state_.target), len(state_.source), best_cost, len(best_state.source), best_state)
    return best_state

