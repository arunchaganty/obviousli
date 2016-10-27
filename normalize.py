"""
Normalizes training data format.
"""

import csv
import sys
import json
from collections import namedtuple
from tqdm import tqdm
import logging

from stanza.nlp.corenlp import AnnotatedSentence
from obviousli.defs import State, Truth

def make_tokens(binary_parse):
    """
    Convert a binary parse: ( ( The men ) ( ( are ( fighting ( outside ( a deli ) ) ) ) . ) )
    to [The men are fighting outside a deli .]
    """
    return binary_parse.replace("(", "").replace(")", "").split()

def make_state(row):
    L = {
        "entailment" : 2,
        "neutral" : 1,
        "contradiction" : 0,
        }
    source = AnnotatedSentence.from_tokens(row.sentence1, make_tokens(row.sentence1_binary_parse))
    target = AnnotatedSentence.from_tokens(row.sentence2, make_tokens(row.sentence2_binary_parse))
    truth = Truth.TRUE
    gold_truth = Truth(L[row.gold_label])
    return State(source, target, truth, gold_truth)

def read_stream(fstream, use_header = True, **kwargs):
    reader = csv.reader(fstream, **kwargs)
    if use_header:
        header = next(reader)
        Row = namedtuple("Row", header)
        return map(lambda row: Row(*row), reader)
    else:
        return reader

def do_snli(args):
    reader = read_stream(args.input,  delimiter='\t')

    for row in tqdm(reader):
        if row.gold_label == "-": continue
        try:
            state = make_state(row)
            args.output.write(json.dumps(state.json) + "\n")
        except AssertionError as e:
            logging.warning(e.args)
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, help="")
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('snli', help='Read the snli dataset')
    command_parser.set_defaults(func=do_snli)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(-1)
    else:
        ARGS.func(ARGS)


# we have tokens, in the same order as sentence. --> I can use this to
# reconstruct before, after and that'd be good enough, no?
# In general, I think it's common to have "here's the full sentence",
# "here's a set of tokens" -> go play!
