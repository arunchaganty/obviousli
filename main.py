#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The obviousli inference system.
"""

import csv
import sys
import json
import logging
import logging.config
from collections import namedtuple

import ipdb
import numpy as np
from tqdm import tqdm

import obviousli.models
from obviousli.defs import AgendaEnvironment, State, Agent, Truth
from obviousli.actions import ActionGenerator, LexicalParaphraseTemplate
from obviousli.util import ConfusionMatrix, grouper


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
    for batch in tqdm(grouper(100, data), total=int(len(data)/100)):
        ys_ = model.predict_on_state_batch(batch)
        ys = (state.gold_truth.value for state in batch)
        for y, y_ in zip(ys, ys_): cm.update(y, np.argmax(y_))
    cm.print_table()
    cm.summary()
    return cm

def get_model_factory(model):
    """import model"""
    return getattr(obviousli.models, model)

def do_model_train(args):
    """
    Train the model specified.
    """
    # [State.new(row.source, row.target, gold_truth=Truth(int(row.gold_truth))) for row in tqdm(list(load_data(args.train_data)))]
    logging.info("Loading training data.")
    train_data = [State.from_json(json.loads(line)) for line in tqdm(list(args.train_data))]
    logging.info("Loading dev data.")
    dev_data = [State.from_json(json.loads(line)) for line in tqdm(list(args.dev_data))]

    # Get model
    Model = get_model_factory(args.entailment_model)
    # Get embedder
    embedder = Model.embedder().construct(train_data + dev_data)
    logging.info("Embedding data.")
    for state in tqdm(train_data + dev_data): 
        rep = embedder.embed(state)
        state.representation = rep[:args.input_length] + [0] * max(0, args.input_length - len(rep))

    model = Model.build(vocab_size=len(embedder.word_map), input_length=args.input_length)
    model.compile(
        optimizer='adagrad',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    logging.info("Model: %s", model.summary())


    for epoch in range(args.n_epochs):
        logging.info("Epoch %d", epoch)

        for state in train_data:
            model.enqueue((state, state.gold_truth))
        model.update()
        model.save(args.model_path)
        # evaluate.
        logging.info("Evaluating model on train...")
        evaluate(model, train_data)
        logging.info("Evaluating model on dev...")
        evaluate(model, dev_data)

def do_model_evaluate(args):
    """
    Evaluate the model specified.
    """
    model = get_model_factory(args.entailment_model).load(args.model_path)
    eval_data = [State.from_json(json.loads(line) for line in tqdm(args.data))]
    logging.info("Evaluating model...")
    evaluate(model, eval_data)

if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--entailment-model', choices=["LexicalCrossUnigramModel"], default="LexicalCrossUnigramModel", help="Which entailment model to use?")
    parser.add_argument('--model_path', default="out", help="Where to load/save models.")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('model-train', help='Train a component model.')
    command_parser.add_argument('--input_length', type=int, default=200, help="Longest accepted sequence")
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
