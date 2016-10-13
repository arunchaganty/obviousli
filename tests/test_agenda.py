#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from obviousli.defs import State, Truth, Agent
from obviousli.defs import Agenda
from obviousli.actions import LexicalParaphraseTemplate, ActionGenerator
from obviousli.agents import GiveUpAgent

SIMPLE_INFERENCES = [
    (State.new("Obama shouted at the speaker.", "Obama screamed at the speaker."), Truth.TRUE),
    (State.new("Cats eat mice.", "Cats eat rats."), Truth.TRUE),
    (State.new("Obama shouted at the speaker.", "Obama cried at the speaker."), Truth.TRUE),
    # TODO: need to handle insertion and truth states.
    #(State.new("Cats eat mice.", "Cats don't eat mice."), Truth.FALSE), # Need to handle insertion
    ]

action_generator = ActionGenerator([
    LexicalParaphraseTemplate("shouted", "screamed"),
    LexicalParaphraseTemplate("shouted", "cried"),
    LexicalParaphraseTemplate("shouted", "moaned"),
    LexicalParaphraseTemplate("screamed", "cried"),
    LexicalParaphraseTemplate("mice", "rats"),
    ])


def test_noop_agenda():
    state = State.new("E=MC2", "Matter and energy are interchangable")
    agent = GiveUpAgent()
    state_ = Agenda(None).run(agent, state)

    assert state_.isEnd()
    assert state_.truth == Truth.NEUTRAL

def test_agenda_with_basic_agent():
    agent = Agent(None)
    agenda = Agenda(None, action_generator)
    for state, gold in SIMPLE_INFERENCES:
        state_ = agenda.run(agent, state)
        assert state_.isEnd()
        assert state_.truth == gold

