"""
Normalizes training data format.
"""

import re
import csv
import sys
import json
import gzip
from xml.etree.ElementTree import ElementTree
from collections import namedtuple
from tqdm import tqdm
import logging

from stanza.nlp.corenlp import AnnotatedSentence
from obviousli.defs import State, Truth
from obviousli.io import GzippableFileType

def make_tokens(binary_parse):
    """
    Convert a binary parse: ( ( The men ) ( ( are ( fighting ( outside ( a deli ) ) ) ) . ) )
    to [The men are fighting outside a deli .]
    @returns tuple
    """
    return tuple(binary_parse.replace("(", "").replace(")", "").split())

def make_pos_tokens(typed_parse):
    """
    Convert typed parse: (ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))
    into joint sequence of (A person is training his horse for a competition .) and (DT NN VBZ VBG PRP$ NN IN DT NN .).

    @returns (tuple, tuple)
    """
    # We only care about the leaves.
    leaves = re.findall(r'\(([^()]+)\)', typed_parse)
    pos, words = zip(*map(str.split, leaves))
    return pos, words

def test_make_pos_tokens():
    pos, words = make_pos_tokens("(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))")
    assert pos == tuple("DT NN VBZ VBG PRP$ NN IN DT NN .".split())
    assert words == tuple("A person is training his horse for a competition .".split())

def make_sentence(text, typed_parse):
    pos, words = make_pos_tokens(typed_parse)
    return AnnotatedSentence.from_tokens(text, words, pos)

def make_state(row, do_annotate=False):
    L = {
        "entailment" : 2,
        "neutral" : 1,
        "contradiction" : 0,
        }

    if do_annotate:
        source = row.sentence1
        target = row.sentence2
        gold_truth = Truth(L[row.gold_label])
        return State.new(source, target, gold_truth)
    else:
        source = make_sentence(row.sentence1, row.sentence1_parse)
        target = make_sentence(row.sentence2, row.sentence2_parse)
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
            state = make_state(row, args.annotate)
            args.output.write(json.dumps(state.json) + "\n")
        except AssertionError as e:
            logging.warning(e.args)
            continue

def do_rte(args):
    tree = ElementTree(file=args.input)
    LABELS = {
        "YES": Truth.TRUE,
        "NO": Truth.FALSE,
        "UNKNOWN": Truth.NEUTRAL,
        }

    for pair in tqdm(tree.findall("pair")):
        assert pair.get("entailment") in LABELS
        label = LABELS[pair.get("entailment")]
        source = pair.findtext("t")
        target = pair.findtext("h")

        try:
            state = State.new(source, target, gold_truth=label)
            args.output.write(json.dumps(state.json) + "\n")
        except AssertionError as e:
            logging.warning(e.args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=GzippableFileType('rt'), default=sys.stdin, help="")
    parser.add_argument('-o', '--output', type=GzippableFileType('wt'), default=sys.stdout, help="")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('snli', help='Read the snli dataset')
    command_parser.add_argument('-a', '--annotate', action="store_true", help="Call the annotation service to actually annotate this data.")
    command_parser.set_defaults(func=do_snli)

    command_parser = subparsers.add_parser('rte', help='Read the RTE dataset')
    command_parser.set_defaults(func=do_rte)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(-1)
    else:
        ARGS.func(ARGS)
