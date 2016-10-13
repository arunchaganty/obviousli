#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obviousli.defs import State, Truth, Agent
from obviousli.actions import GiveUpAction
from obviousli.agents import GiveUpAgent

def test_default_agent():
    state = State.new("E=MC2", "Matter and energy are interchangable")
    agent = Agent(None)
    actions = agent.rerank(state, [GiveUpAction()])

    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, GiveUpAction)

def test_give_up_agent():
    state = State.new("E=MC2", "Matter and energy are interchangable")
    agent = GiveUpAgent()
    actions = agent.rerank(state, [])

    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, GiveUpAction)

