#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitions of standard actions
"""

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

# TODO(chaganty): implement the following actions.
#     - Guess
#     - LexicalParaphrase

#     - SyntacticParaphrase
#     - SyntacticTransformation
