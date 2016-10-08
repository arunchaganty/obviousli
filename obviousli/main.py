#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The obviousli inference system.
"""

import csv
import sys

def do_command(args):
    reader = csv.reader(args.input, delimiter='\t')
    writer = csv.writer(args.output, delimiter='\t')

    header = next(reader)
    assert len(header) > 0, "Invalid header"

    writer.writerow(header)
    for row in reader:
        writer.writerow(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=argparse.FileType('r'), default=sys.stdin, help="")
    parser.add_argument('--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
