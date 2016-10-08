#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obviousli.defs import State, Truth
from obviousli.actions import GiveUpAction
from obviousli.agents import GiveUpAgent

def test_give_up_agent():
    state = State("E=MC2", "Matter and energy are interchangable", Truth.TRUE, None)
    agent = GiveUpAgent()
    action = agent.act(state)

    assert isinstance(action, GiveUpAction)
