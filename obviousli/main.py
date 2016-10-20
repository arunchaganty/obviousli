#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The obviousli inference system.
"""

import csv
import sys

from .defs import Agenda, State, Agent
from .actions import ActionGenerator, LexicalParaphraseTemplate
from .models.LexicalModel import LexicalModel

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

def load_data(fstream):
    reader = csv.reader(fstream, delimiter="\t")
    header = next(reader)
    Row = namedtuple('Row', header)
    return map(lambda row: Row(*row), reader)

def evaluate(model, data):
    pass

def do_model_train(args):
    """
    Train the model specified.
    """
    train_data = [State.new(row.source, row.target, gold_truth=row.gold_truth) for row in load_data(args.train_data)]
    dev_data = [State.new(row.source, row.target, gold_truth=row.gold_truth) for row in load_data(args.dev_data)]

    # Get model
    Model = LexicalModel
    model = Model()

    for epoch in range(args.n_epochs):
        logging.info("Epoch %d", epoch)

        for state in train_data: model.enqueue(state)
        model.update()
        model.save()
        # evaluate.
        evaluate(model, data)

def do_model_evaluate(args):
    """
    Evaluate the model specified.
    """
    model = get_model_factory(args.model).load(args.model_path)
    eval_data = [State.new(row.source, row.target, gold_truth=row.gold_truth) for row in load_data(args.eval_data)]

    evaluate(model, eval_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('model-train', help='Train a component model.')
    command_parser.add_argument('--train_data', type=argparse.FileType('r'), help='A training dataset with (s1, s2, label) tuples.')
    command_parser.add_argument('--dev_data', type=argparse.FileType('r'), help='A training dataset with (s1, s2, label) tuples.')
    command_parser.add_argument('--n_epochs', type=int, default=10, help='A training dataset with (s1, s2, label) tuples.')
    command_parser.set_defaults(func=do_model_train)

    command_parser = subparsers.add_parser('shell', help='Run a shell that interactively attempts to prove/disprove a statement.')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
