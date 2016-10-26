#!/usr/bin/env python3
# -*- coding: utf-7 -*-
"""
Top-level definitions.
"""
import heapq
import logging
from enum import Enum
from abc import ABC, abstractmethod

from stanza.nlp.corenlp import CoreNLPClient, AnnotatedSentence

import obviousli.config as config

class Truth(Enum):
    FALSE=0
    NEUTRAL=1
    TRUE=2

    def sym(self):
        """
        Symbolic representation
        """
        if self == Truth.TRUE:
            return '*'
        elif self == Truth.NEUTRAL:
            return '?'
        else:
            return '!'

    def __lt__(self, other):
        return self.value < other.value

class State(object):
    """
    Represents inference state, with source and target sentences and arbitrary representation.
    """
    _client = CoreNLPClient(server=config.CORENLP_SERVER, default_annotators=config.CORENLP_ANNOTATORS)

    def __init__(self, source, target, truth, gold_truth=None, previous_state_action=None):
        self.source = source
        self.target = target
        self.truth = truth
        self.gold_truth = gold_truth
        self.previous_state_action = previous_state_action # to maintain a backpointer.

    def __str__(self):
        return "{} -> {}{}".format(self.source, self.truth.sym(), self.target)

    def __repr__(self):
        return "[S: {}]".format(str(self))

    def __lt__(self, other):
        if self.source != other.source:
            return self.source < other.source
        elif self.target != other.target:
            return self.target < other.target
        else:
            return self.truth < other.truth

    @classmethod
    def from_json(cls, obj):
        source = AnnotatedSentence.from_json(obj["source"])
        target = AnnotatedSentence.from_json(obj["target"])
        truth = Truth(obj["truth"])
        gold_truth = obj["gold_truth"] and Truth(obj["gold_truth"])
        # TODO(chaganty): how to serialize previous_state_action?
        previous_state_action = None
        return State(source, target, truth, gold_truth, previous_state_action)

    @property
    def representation(self):
        # TODO: generate these representations.
        return None

    def isEnd(self):
        """
        @returns: true iff the source sentence is equivalent to the target.
        """
        return self.source == self.target

    @classmethod
    def new(cls, source, target, gold_truth=None):
        source = cls._client.annotate(source)[0]
        target = cls._client.annotate(target)[0]
        assert isinstance(source, AnnotatedSentence)
        assert isinstance(target, AnnotatedSentence)
        return State(source, target, Truth.TRUE, gold_truth=gold_truth)

    def replace(self, 
                source=None,
                target=None,
                truth=None,
                gold_truth=None,
                previous_state_action=None):
        return State(
            source or self.source,
            target or self.target,
            truth or self.truth,
            gold_truth or self.gold_truth,
            previous_state_action or self.previous_state_action)

class Action(ABC):
    """
    An action tranforms one state to another state.
    """

    def __init__(self, representation):
        self.representation = representation

    @abstractmethod
    def __call__(self, state):
        """
        Transform input state.
        """
        pass

    def __str__(self):
        return "{}".format(id(self))

    def __repr__(self):
        return "[{}: {}]".format(self.__class__.__name__, str(self))

class Agent(object):
    """
    A reasoning agent.
        - receives current state and previous trajectory segment
        - a trajectory segment consists of the previous state, action taken and reward signal (the latter usually
          comes from critic).
        - uses actor model to choose top-k actions.
        - appends trajectory segment to the actor model's training history.
    """
    def __init__(self, actor_model, top_k=1):
        self.actor_model = actor_model
        self.top_k = top_k

    def act(self, state, valid_actions):
        """
        @state: current state
        @valid_actions: a list of valid actions that can be taken in @state
        @returns: @self.top_k (score, action)s
        """
        score_of = lambda action: self.actor_model.predict((state, action))
        return sorted(((score_of(action), action) for action in valid_actions), reverse=True)[:self.top_k]

    def incorporate_feedback(self, state, action, reward):
        self.actor_model.enqueue(((state, action), reward))

class AgendaEnvironment(ABC):
    """
    An environment:
        - produces action candidates
        - updates state given an action
        - provides (raw) reward
        - updates critic model with reward to score states
        - maintains an agenda (ordered by critic)
        - sends actions and reward signals to agent.
    """

    def __init__(self, agent, critic_model, action_generator=lambda _: [], reward_fn = lambda _: 0., gamma=0.9):
        self.agent = agent
        self.critic_model = critic_model
        self.action_generator = action_generator
        self.gamma = gamma
        self.reward_fn = reward_fn
        self.logger = logging.getLogger('obviousli')

    def run_episode(self, state):
        """
        Allows the agent to run on the state

        @state: starting state
        @return: final state returned by the agent
        """
        if state.isEnd(): return state

        agenda = [(0, state)]
        while len(agenda) > 0:
            state = self._run_step(agenda)
            if state.isEnd(): return state
        raise Exception("Agenda completed without arriving at end state")

    def _run_step(self, agenda):
        """
        In a single step,
            - the "best" state in the agenda is popped
            - the agent chooses the next top_k actions to take
            - the corresponding next states are added to the agenda
            - feedback from the reward is added to the training queues
              of the actor and critic models.
        """
        # log top of agenda
        value, state = heapq.heappop(agenda)
        value = -value # invert
        self.logger.info("Visiting state: (%.2f) %s", value, state)
        if not state.isEnd():
            for _, action in self.agent.act(state, self.action_generator(state)):
                # Apply action on state and save on queue.
                state_ = action(state)
                value_ = self.critic_model.predict(state_)
                heapq.heappush(agenda, (-value_,state_)) # inverting value because of min-heap

                # Update actor and critic models.
                reward = self.reward_fn(state_)
                self.logger.info("Adding to agenda: (%.2f) %s -> %s (reward=%.2f)", value_, action, state_, reward)

                reward_ = value_
                reward_ += self.gamma * reward
                self.critic_model.enqueue(((state, action, state_, value, reward), reward_))
                self.agent.incorporate_feedback(state, action, value_ - value) # this is the TD-update
        return state

