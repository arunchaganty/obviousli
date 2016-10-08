#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obviousli.defs import State, Truth
from obviousli.defs import Agenda
from obviousli.agents import GiveUpAgent

def test_noop_agenda():
    state = State("E=MC2", "Matter and energy are interchangable", Truth.TRUE, None)
    agent = GiveUpAgent()
    state_ = Agenda([]).run(agent, state)

    assert state_.isEnd()
    assert state_.truth == Truth.NEUTRAL

