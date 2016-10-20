#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple inferences
"""

from obviousli.defs import State, Truth, Agent, AgendaEnvironment
from obviousli.models import ActorModel, CriticModel
from obviousli.actions import GiveUpAction, ActionGenerator, LexicalParaphraseTemplate
from obviousli.util import normalized_edit_distance

class MockActorModel(ActorModel):
    def __init__(self):
        super(MockActorModel, self).__init__(input=[], output=[])
        
    def predict(self, state, action):
        if isinstance(action, GiveUpAction):
            return -1.
        else:
            return 0.

    def update(self):
        return

class MockCriticModel(CriticModel):
    def __init__(self):
        super(MockCriticModel, self).__init__(input=[], output=[])

    def predict(self, state):
        if state.truth == Truth.NEUTRAL:
            return -1.
        else:
            return -normalized_edit_distance(state.source, state.target)
    def update(self):
        return

def mock_reward_fn(state):
    """
    Reward signal.
    -.1 for every step taken.
    +1 for reaching the correct end state.
    +(1-dist) for levenstein distance between the two.
    -1 for reaching wrong end state.
    """
    ret = -0.1
    if state.isEnd():
        if state.gold_truth is not None and state.truth == state.gold_truth:
            ret += 1
        elif state.gold_truth is not None and state.truth != state.gold_truth:
            ret += -1
        else:
            ret += 0.5
    else:
        pass
    return ret

SIMPLE_INFERENCES = [
    State.new("Obama shouted at the speaker.", "Obama screamed at the speaker.", Truth.TRUE),
    State.new("Cats eat mice.", "Cats eat rats.", Truth.TRUE),
    State.new("Obama shouted at the speaker.", "Obama cried at the speaker.", Truth.TRUE),
    # TODO(chaganty): need to handle insertion and truth states.
    #(State.new("Cats eat mice.", "Cats don't eat mice."), Truth.FALSE), # Need to handle insertion
    ]

action_generator = ActionGenerator([
    LexicalParaphraseTemplate("shouted", "screamed"),
    LexicalParaphraseTemplate("shouted", "moaned"),
    LexicalParaphraseTemplate("screamed", "cried"),
    LexicalParaphraseTemplate("mice", "rats"),
    ],[
        GiveUpAction()
    ])

def test_agenda_with_multiple_options():
    TOP_K = 4
    agent = Agent(MockActorModel(), top_k=TOP_K)
    agenda = AgendaEnvironment(
        agent,
        MockCriticModel(),
        action_generator=action_generator,
        reward_fn=mock_reward_fn)

    state = State.new("Obama shouted at the speaker.", "Obama screamed at the speaker.", Truth.TRUE)
    queue = [(0,state)]
    agenda._run_step(queue)

    assert len(queue) == TOP_K
    assert len(set(id(st) for st in queue)) == TOP_K # Make sure the returned states are actually new.
    assert not isinstance(queue[0], GiveUpAction)

def test_agenda_with_basic_agent():
    agent = Agent(MockActorModel(), top_k=3)
    agenda = AgendaEnvironment(
        agent,
        MockCriticModel(),
        action_generator=action_generator,
        reward_fn=mock_reward_fn)

    for state in SIMPLE_INFERENCES:
        state_ = agenda.run_episode(state)
        assert state_.isEnd()
        assert state_.truth == state_.gold_truth

