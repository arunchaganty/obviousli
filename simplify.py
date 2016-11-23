#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import logging
import logging.config
import heapq
import json

from tqdm import tqdm

import obviousli.nlp as on
from obviousli.io import GzippableFileType, load_states_from_file

logger = logging.getLogger('obviousli')
logger.setLevel(logging.INFO)


# measure overlap by looking at how many NNPs and lemmas match between
# sentences.
# As many NNPs as possible should match, and choose the one with the largest lemma
# overlap.

def score_mod(source, target):
    """
    Score based on Jaccard score of lemmas
    """
    s = set(source.lemmas)
    t = set(target.lemmas)

    return len(s.intersection(t))/len(s.union(t))

def keep_token(t):
    return t.pos.startswith("NN") or t.pos.startswith("JJ") or t.pos.startswith("CD")

def filter_mod(source, target, original_source):
    """
    Make sure that s, t pair have as many NNPs in common as original source.
    """
    nnps_original = set([t.lemma for t in original_source if keep_token(t)])
    nnps_source = set([t.lemma for t in source if keep_token(t)])
    nnps_target = set([t.lemma for t in target if keep_token(t)])

    return nnps_original.intersection(nnps_target) == nnps_source.intersection(nnps_target)

def mod_gen(sent):
    dp = sent.depparse()
    for label in ["ccomp", "nmod", "amod", "advmod", "ccomp", "acl", "appos", "advcl"]:
        for (_, j, _) in on.find_edges(dp, label=label):
            yield on.drop_subtree(dp, j)
            yield on.keep_subtree(dp, j)

def simplify(state_):
    """ Deterministic modification -- recursively simply until you get a
        sentence with all the named entities between the two entences."""

    candidates = [(-score_mod(state_.source, state_.target), state_)]
    visited = set([str(state_.source)])

    best_cost, best_state = 0, None
    while len(candidates) > 0:
        cost, state = heapq.heappop(candidates)
        logger.debug("modifying %.2f %s", cost, state)

        # update best tracker
        if cost < best_cost:
            best_cost, best_state = cost, state

        # simplify the state.source
        for mod in mod_gen(state.source):
            if filter_mod(mod, state.target, state_.source):
                cost, state = -score_mod(mod, state.target), state.replace(source=mod)
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
    logger.info("best pair %.2f %d -> %.2f %d %s", -score_mod(state_.source, state_.target), len(state_.source), best_cost, len(best_state.source), best_state)
    return best_state

def do_command(args):
    for state in tqdm(load_states_from_file(args.input)):
        state_ = simplify(state)
        args.output.write(json.dumps(state_.json) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=GzippableFileType('rt'), default=sys.stdin, help="")
    parser.add_argument('-o', '--output', type=GzippableFileType('wt'), default=sys.stdout, help="")
    parser.add_argument('--config', type=argparse.FileType('r'), default="logging.json", help="Path to logging configuration file.")
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        logging.config.dictConfig(json.load(ARGS.config))
        ARGS.func(ARGS)

