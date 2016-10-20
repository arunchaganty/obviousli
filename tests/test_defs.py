#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test aspects of definitions.
"""

from obviousli.defs import State, Truth, Agent, AgendaEnvironment
from obviousli.models import ActorModel, CriticModel
from obviousli.actions import GiveUpAction, ActionGenerator, LexicalParaphraseTemplate
from obviousli.util import edit_distance

class MockActorModel(ActorModel):
    def __init__(self):
        super(MockActorModel, self).__init__(input=[], output=[])

    def predict(self, x):
        state, action = x
        return edit_distance(state.source, state.target)

    def update(self):
        return

class MockCriticModel(CriticModel):
    def __init__(self):
        super(MockCriticModel, self).__init__(input=[], output=[])
    def predict(self, example):
        return 0.
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

def test_state():
    state = State.new("E=MC2", "Matter and energy are interchangable")
    assert state.truth == Truth.TRUE
    assert not state.isEnd()
    state = State.new("E=MC2", "E=MC2")
    assert state.isEnd()
    assert state.truth == Truth.TRUE

def test_agent_act():
    agent = Agent(MockActorModel(), top_k=1)

    state = State.new("E=MC2", "Matter and energy are interchangable")
    actions = agent.act(state, [GiveUpAction()])
    assert len(actions) == 1
    _, action = actions[0]
    assert isinstance(action, GiveUpAction)

def test_agent_feedback():
    agent = Agent(MockActorModel(), top_k=1)

    state = State.new("E=MC2", "Matter and energy are interchangable")
    action = GiveUpAction()
    reward = 0
    agent.incorporate_feedback(state, action, reward)
    assert len(agent.actor_model.queue) == 1
    segment = agent.actor_model.queue[0]
    assert len(segment) == 2
    (state_, action_), reward_ = segment
    assert state_ == state
    assert action_ == action
    assert reward_ == reward

def test_agenda_environment():
    agent = Agent(MockActorModel(), top_k=1)
    agenda = AgendaEnvironment(
        agent,
        MockCriticModel(),
        action_generator=lambda _: [GiveUpAction()],
        reward_fn=mock_reward_fn)

    state = State.new("E=MC2", "Matter and energy are interchangable", Truth.TRUE)
    state_ = agenda.run_episode(state)

    assert state_.isEnd()
    assert state_.truth == Truth.NEUTRAL
    assert len(agent.actor_model.queue) == 1
    _, reward = agent.actor_model.queue[0]
    assert reward == 0 # Reward here is still 0

    assert len(agenda.critic_model.queue) == 1
    _, reward_ = agenda.critic_model.queue[0]
    assert reward_ < 0
