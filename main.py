#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The obviousli inference system.
"""

import csv
import sys
from importlib import import_module
from collections import namedtuple
import logging
import logging.config

import ipdb
import numpy as np
from tqdm import tqdm

from obviousli.defs import AgendaEnvironment, State, Agent, Truth
from obviousli.actions import ActionGenerator, LexicalParaphraseTemplate
from obviousli.models.LexicalModel import LexicalModel
from obviousli.util import ConfusionMatrix


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
    pass
#    action_generator = ActionGenerator([
#        LexicalParaphraseTemplate("shouted", "screamed"),
#        LexicalParaphraseTemplate("shouted", "cried"),
#        LexicalParaphraseTemplate("shouted", "moaned"),
#        LexicalParaphraseTemplate("screamed", "cried"),
#        LexicalParaphraseTemplate("mice", "rats"),
#        ])
#    agent = Agent(None)
#    agenda = AgendaEnvironment(agent, None, action_generator)
#
#    while True:
#        try:
#            source = input("$s> ")
#            if len(source) == 0: continue
#            target = input("$t> ")
#            if len(target) == 0: continue
#            state = State.new(source, target)
#
#            gen = agenda.run_episode(state)
#            while True:
#                try:
#                    queue = next(gen)
#                    printq(queue)
#                except StopIteration as e:
#                    state_ = e.value
#            print("$o> {}".format(state_.truth))
#        except EOFError:
#            pass

def load_data(fstream):
    reader = csv.reader(fstream, delimiter="\t")
    header = next(reader)
    Row = namedtuple('Row', header)
    return map(lambda row: Row(*row), reader)

def evaluate(model, data):
    cm = ConfusionMatrix([t.name for t in Truth])
    for state in data:
        y_ = model.predict(state)
        cm.update(state.gold_truth.value, np.argmax(y_))
    cm.print_table()
    cm.summary()
    return cm

def get_model_factory(model):
    """import model"""
    return getattr(import_module('obviousli.models.{0}'.format(model)), model)

def do_model_train(args):
    """
    Train the model specified.
    """
    train_data = [State.new(row.source, row.target, gold_truth=Truth(int(row.gold_truth))) for row in tqdm(list(load_data(args.train_data)))]
    dev_data = [State.new(row.source, row.target, gold_truth=Truth(int(row.gold_truth))) for row in tqdm(list(load_data(args.dev_data)))]

    words = set([])
    for state in train_data:
        words.update(tok.word for tok in state.source.tokens)
        words.update(tok.word for tok in state.target.tokens)
    for state in dev_data:
        words.update(tok.word for tok in state.source.tokens)
        words.update(tok.word for tok in state.target.tokens)

    word_map = {word : i for i, word in enumerate(sorted(words))}

    # Get model
    Model = get_model_factory(args.entailment_model)
    model = Model.build(word_map)
    model.compile(
        optimizer='adagrad',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    for epoch in range(args.n_epochs):
        logging.info("Epoch %d", epoch)

        for state in train_data:
            model.enqueue((state, state.gold_truth))
        model.update()
        model.save(args.model_path)
        # evaluate.
        evaluate(model, train_data)
        evaluate(model, dev_data)

def do_model_evaluate(args):
    """
    Evaluate the model specified.
    """
    model = get_model_factory(args.entailment_model).load(args.model_path)
    eval_data = [State.new(row.source, row.target, gold_truth=row.gold_truth) for row in tqdm(list(load_data(args.eval_data)))]

    evaluate(model, eval_data)


logging.config.fileConfig('logging_config.ini')
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--entailment-model', choices=["LexicalModel"], default="LexicalModel", help="Which entailment model to use?")
    parser.add_argument('--model_path', default="out", help="Where to load/save models.")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('model-train', help='Train a component model.')
    command_parser.add_argument('--train_data', type=argparse.FileType('r'), required=True, help='A training dataset with (s1, s2, label) tuples.')
    command_parser.add_argument('--dev_data', type=argparse.FileType('r'), required=True, help='A training dataset with (s1, s2, label) tuples.')
    command_parser.add_argument('--n_epochs', type=int, default=10, help='A training dataset with (s1, s2, label) tuples.')
    command_parser.set_defaults(func=do_model_train)

    command_parser = subparsers.add_parser('shell', help='Run a shell that interactively attempts to prove/disprove a statement.')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
