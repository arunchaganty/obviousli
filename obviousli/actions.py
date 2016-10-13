#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitions of standard actions
"""

from abc import abstractmethod
from .defs import State, Action, Truth

class GiveUpAction(Action):
    """
    Do a single thing: give up.
    """

    def __init__(self):
        super(GiveUpAction, self).__init__(None)

    def __call__(self, state):
        state.source = state.target
        state.truth = Truth.NEUTRAL
        return state

    def __str__(self):
        return "[GiveUp()]"

class ActionTemplate(object):
    """
    Can be applied on a state to generate actions.
    """

    @abstractmethod
    def generate(self, state):
        """
        Generate all possible applications of this template to the state.
        """
        return ()

# TODO(chaganty): implement the following actions.
#     - Guess
#     - LexicalParaphrase
class LexicalParaphrase(Action):
    """
    Replace a lexical sequence.
    """
    def __init__(self, input_, output, apply_on_source=True, match_idx=None):
        super(LexicalParaphrase, self).__init__(None)
        self.input = input_
        self.output = output
        self.apply_on_source = apply_on_source
        self.match_idx = match_idx

    def __str__(self):
        return "[LexicalParaphrase({}:{}@{} -> {})]".format("S" if self.apply_on_source else "T", self.input, self.match_idx, self.output)

    def __call__(self, state):
        if self.apply_on_source:
            state.source = state.source[:self.match_idx] + self.output + state.source[self.match_idx + len(self.input):]
        else:
            state.target = state.target[:self.match_idx] + self.output + state.target[self.match_idx + len(self.input):]
        return state

class LexicalParaphraseTemplate(ActionTemplate):
    """
    Generates LexicalParaphrase actions for a given 'input'/'output' pair.
    """

    def __init__(self, input_, output):
        self.input = input_
        self.output = output

    def generate(self, state):
        # check if input is in the source.
        start = 0
        while state.source.find(self.input, start) > -1:
            idx = state.source.find(self.input, start)
            yield LexicalParaphrase(self.input, self.output, apply_on_source=True, match_idx=idx)
            start = idx+1

        start = 0
        while state.target.find(self.input, start) > -1:
            idx = state.target.find(self.input, start)
            yield LexicalParaphrase(self.input, self.output, apply_on_source=False, match_idx=idx)
            start = idx+1

# - PhraseParaphrase
