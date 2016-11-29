#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import logging
import logging.config
import json

from tqdm import tqdm

import obviousli.nlp as on
from obviousli.io import GzippableFileType, load_states_from_file

logger = logging.getLogger('obviousli')
logger.setLevel(logging.INFO)

def do_command(args):
    for state in tqdm(load_states_from_file(args.input)):
        state_ = on.simplify(state)
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

