#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A lexical baseline model as described in:
> Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large
> annotated corpus for learning natural language inference. Proceedings of
> the 2015 Conference on Empirical Methods in Natural Language
> Processing,Lisbon, Portugal, 17-21 September 2015, (September),
> 632–642.
"""

import os
import sys
import json
import pickle
import logging
import logging.config
from numpy import zeros
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .defs import State, Truth
from .embeddings import LexicalBaselineEmbedder
from .util import ConfusionMatrix

def make_matrix(embedder, data):
    N, D = len(data), len(embedder.features)
    X_I = []
    X_J = []
    X_V = []
    y = zeros((N,))
    for i, datum in enumerate(data):
        j, v = zip(*embedder.embed(datum))
        X_I += [i] * len(j)
        X_J += j
        X_V += v
        y[i] = datum.gold_truth.value
    return coo_matrix((X_V,(X_I,X_J)),shape=(N,D)), y

def evaluate(model, X, ys):
    cm = ConfusionMatrix([t.name for t in Truth])
    ys_ = model.predict(X)
    for y, y_ in zip(ys, ys_): cm.update(y, y_)
    cm.print_table()
    cm.summary()
    return cm

def do_train(args):
    """
    Train the model specified.
    """
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    logging.info("Loading training data.")
    train_data = [State.from_json(json.loads(line)) for line in tqdm(list(args.data_train))]
    logging.info("Loading %d instances.", len(train_data))
    logging.info("Loading dev data.")
    dev_data = [State.from_json(json.loads(line)) for line in tqdm(list(args.data_dev))]
    logging.info("Loading %d instances.", len(dev_data))

    # Get embedder
    logging.info("Building feature map.")
    embedder = LexicalBaselineEmbedder.construct(data=train_data)
    logging.info("Built with %d features.", len(embedder.features))
    embedder.save(os.path.join(args.model_path, 'features.txt'))

    # Get model
    model = LogisticRegression(C=1, penalty='l1', tol=0.01)

    # Embed data.
    logging.info("Embedding data.")
    X_train, y_train = make_matrix(embedder, train_data)
    X_dev, y_dev = make_matrix(embedder, dev_data)

    logging.info("Training model.")
    model.fit(X_train, y_train)
    with open(os.path.join(args.model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logging.info("Training error.")
    evaluate(model, X_train, y_train)
    logging.info("Dev error.")
    evaluate(model, X_dev, y_dev)


def do_evaluate(args):
    """
    Evaluate the model specified.
    """
    logging.info("Loading embedding.")
    embedder = LexicalBaselineEmbedder.load(os.path.join(args.model_path, "features.txt"))
    logging.info("Loaded embedding with %d features.", len(embedder.features))

    logging.info("Loading model.")
    with open(os.path.join(args.model_path, "model.pkl"), "r") as f:
        model = pickle.load(f)

    logging.info("Loading data.")
    data = [State.from_json(json.loads(line)) for line in tqdm(list(args.data))]
    logging.info("Loading %d instances.", len(data))

    logging.info("Embedding data.")
    X, y = make_matrix(embedder, data)

    logging.info("Error.")
    evaluate(model, X, y)

    # TODO: print output predictions with features


def do_shell(args):
    pass

if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    import argparse
    parser = argparse.ArgumentParser(description='obviousli.lexical_baseline: a lexical baseline model.')
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('train', help='Train model.')
    command_parser.add_argument('-m', '--model-path', default="state/lexical_baseline/", help="directory path-root to load/save models")
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), required=True, help='A training dataset with (s1, s2, label) tuples.')
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), required=True, help='A dev dataset with (s1, s2, label) tuples.')
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='Evaluate model.')
    command_parser.add_argument('-m', '--model-path', default="state/lexical_baseline/", help="directory path-root to load/save models")
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), required=True, help='A dataset with (s1, s2, label) tuples.')
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='Run a shell that interactively attempts to prove/disprove a statement.')
    command_parser.add_argument('-m', '--model-path', default="state/lexical_baseline/", help="directory path-root to load/save models")
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
