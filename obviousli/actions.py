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

    def __lt__(self, other):
        return isinstance(other, GiveUpAction)

    def __call__(self, state):
        return state.replace(source=state.target, truth=Truth.NEUTRAL, previous_state_action=(state, self))

    def __str__(self):
        return "GiveUp()"

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

class ActionGenerator(object):
    """
    The action generator is responsible for generating all possible
    actions given resources.
    """

    def __init__(self, templates, fixed_actions=None):
        #: A sequence of action templates
        self._templates = templates
        self._fixed_actions = fixed_actions

    def __call__(self, state):
        for template in self._templates:
            yield from template.generate(state)
        if self._fixed_actions:
            yield from self._fixed_actions

# TODO(chaganty): implement the following actions.
#     - Guess
# TODO(chaganty): allow the gradient to pass into the Guess action.

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
        return "{}:{}@{} -> {}".format("S" if self.apply_on_source else "T", self.input, self.match_idx, self.output)

    def __lt__(self, other):
        if isinstance(other, LexicalParaphrase):
            return self.input < other.input
        else:
            return True

    def __call__(self, state):
        if self.apply_on_source:
            new_ = state.source[:self.match_idx] + self.output + state.source[self.match_idx + len(self.input):]
            return state.replace(source=new_, previous_state_action=(state, self))
        else:
            new_ = state.target[:self.match_idx] + self.output + state.target[self.match_idx + len(self.input):]
            return state.replace(target=new_, previous_state_action=(state, self))
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
