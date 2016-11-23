#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split examples into train, dev (stratified).
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import logging
import logging.config
import json
import random
from collections import defaultdict

from tqdm import tqdm

from obviousli.io import GzippableFileType, load_states_from_file

logger = logging.getLogger('obviousli')
logger.setLevel(logging.INFO)

def do_command(args):
    random.seed(args.seed)

    # Bin exaxmples by type
    per_label = defaultdict(list)
    for state in tqdm(load_states_from_file(args.input)):
        per_label[state.gold_truth].append(state)

    # Now shuffle each class by proportion, split by fraction, shuffle again and save.
    train = []
    dev = []
    for grp in per_label.values():
        random.shuffle(grp)
        split = int(len(grp) * args.split)
        train += grp[:split]
        dev += grp[split:]

    random.shuffle(train)
    random.shuffle(dev)

    for state in train:
        args.output_train.write(json.dumps(state.json) + "\n")
    for state in dev:
        args.output_dev.write(json.dumps(state.json) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', type=GzippableFileType('rt'), default=sys.stdin, help="")
    parser.add_argument('-ot', '--output-train', type=GzippableFileType('wt'), default=sys.stdout, help="")
    parser.add_argument('-od', '--output-dev', type=GzippableFileType('wt'), default=sys.stdout, help="")
    parser.add_argument('-s', '--split', type=float, default=0.8, help="Fraction of examples to split for train")
    parser.add_argument('--config', type=argparse.FileType('r'), default="logging.json", help="Path to logging configuration file.")
    parser.add_argument('--seed', type=int, default=42, help="Seed to initalize randomness with.")
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

