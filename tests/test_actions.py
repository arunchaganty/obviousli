#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obviousli.defs import State, Truth
from obviousli.actions import GiveUpAction, LexicalParaphrase, LexicalParaphraseTemplate

def test_give_up_action():
    state = State("E=MC^2", "Matter and energy are interchangable", Truth.TRUE, None)
    action = GiveUpAction()
    state_ = action(state)

    assert state_.source == "Matter and energy are interchangable"
    assert state_.isEnd()
    assert state_.truth == Truth.NEUTRAL

def test_lexical_paraphrase():
    state = State("Obama shouted at the speaker.", "Obama screamed at the speaker.", Truth.TRUE, None)
    action = LexicalParaphrase("shouted", "screamed", True, 6)
    state_ = action(state)

    assert state_.source == "Obama screamed at the speaker."
    assert state_.isEnd()
    assert state.truth == Truth.TRUE

def test_lexical_paraphrase_template():
    state = State("Obama shouted at the speaker.", "Obama screamed at the speaker.", Truth.TRUE, None)
    template = LexicalParaphraseTemplate("shouted", "screamed")
    actions = list(template.generate(state))

    assert len(actions) == 1
    action = actions[0]
    assert action.input == "shouted"
    assert action.output == "screamed"
    assert action.apply_on_source
    assert action.match_idx == 6
