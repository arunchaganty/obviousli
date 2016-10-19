#!/usr/bin/env python3
# -*- coding: utf-7 -*-
"""
Top-level definitions.
"""
from enum import Enum
import heapq
from abc import ABC, abstractmethod
import logging

class Truth(Enum):
    FALSE=-1
    NEUTRAL=0
    TRUE=1

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

    def __init__(self, source, target, truth, representation, gold_truth=None, previous_state_action=None):
        self.source = source
        self.target = target
        self.truth = truth
        # TODO: we'll be generating these representations soon...
        self.representation = representation
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

    def isEnd(self):
        """
        @returns: true iff the source sentence is equivalent to the target.
        """
        return self.source == self.target

    @classmethod
    def new(cls, source, target, gold_truth=None):
        return State(source, target, Truth.TRUE, None, gold_truth=gold_truth)

    def replace(self, 
                source=None,
                target=None,
                truth=None,
                representation=None,
                gold_truth=None,
                previous_state_action=None):
        return State(
            source or self.source,
            target or self.target,
            truth or self.truth,
            representation or self.representation,
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

class Model(ABC):
    """
    Types (generic): I -> J
    A model has two views: a test and train view.
    Test: a model predicts probability of type J given an element of type I.
    Train: a model stores a list of training examples that can be enqueued by anyone.
           Furthermore, the model can be "updated" to learn on these training examples.
    """
    def __init__(self):
        self.queue = []

    @abstractmethod
    def predict(self, example):
        """
        @example - a d-dimensional vector input to the model.
        @returns a distribution over outputs.
        """
        pass

    def enqueue(self, example):
        """
        Appends @example to list of elements to be queued.
        @example - a tuple of (input, output)
        """
        self.queue.append(example)

    @abstractmethod
    def update(self):
        """
        Updates parameters of model.
        """
        pass

class ActorModel(Model):
    """
    Type signature: state, action -> value 
    """
    def predict(self, example):
        state, action = example
        return self._predict(state, action)

    @abstractmethod
    def _predict(self, state, action):
        pass

    def enqueue(self, example):
        """
        @example: is a ((state, action), reward_signal) triple.
        """
        assert len(example) == 2 and len(example[0]) == 2, "invalid training example"
        super(ActorModel, self).enqueue(example)

class CriticModel(Model):
    """
    Type signature: state -> value.
    """
    pass

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

                self.critic_model.enqueue(((state, value), (reward + self.gamma*value_)))
                self.agent.incorporate_feedback(state, action, value_ - value) # this is the TD-update
        return state

