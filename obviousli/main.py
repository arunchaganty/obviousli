#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The obviousli inference system.
"""

import csv
import sys

from .defs import Agenda, State, Agent
from .actions import ActionGenerator, LexicalParaphraseTemplate

def printq(queue):
    for score, (state, action) in queue:
        print("{}\t{}\t{}".format(score, state, action))

def do_command(args):
    reader = csv.reader(args.input, delimiter='\t')
    writer = csv.writer(args.output, delimiter='\t')

    header = next(reader)
    assert len(header) > 0, "Invalid header"

    writer.writerow(header)
    for row in reader:
        writer.writerow(row)

def do_shell(args):
    action_generator = ActionGenerator([
        LexicalParaphraseTemplate("shouted", "screamed"),
        LexicalParaphraseTemplate("shouted", "cried"),
        LexicalParaphraseTemplate("shouted", "moaned"),
        LexicalParaphraseTemplate("screamed", "cried"),
        LexicalParaphraseTemplate("mice", "rats"),
        ])
    agenda = Agenda(None, action_generator)
    agent = Agent(None)

    while True:
        try:
            source = input("$s> ")
            if len(source) == 0: continue
            target = input("$t> ")
            if len(target) == 0: continue
            state = State.new(source, target)

            gen = agenda.run(agent, state)
            while True:
                try:
                    queue = next(gen)
                    printq(queue)
                except StopIteration as e:
                    state_ = e.value
            print("$o> {}".format(state_.truth))
        except EOFError:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('shell', help='Run a shell that interactively attempts to prove/disprove a statement.')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
