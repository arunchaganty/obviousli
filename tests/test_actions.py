#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obviousli.defs import State, Truth
from obviousli.actions import GiveUpAction

def test_give_up_action():
    state = State("E=MC2", "Matter and energy are interchangable", Truth.TRUE, None)
    action = GiveUpAction()
    state_ = action(state)

    assert state_.source == "Matter and energy are interchangable"
    assert state_.isEnd()

