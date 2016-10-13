#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-level definitions.
"""
from enum import Enum
import heapq
from abc import ABC, abstractmethod

class Truth(Enum):
    FALSE=-1
    NEUTRAL=0
    TRUE=1

    def sym(self):
        """
        Symbolic representation
        """
        if self.value == Truth.TRUE:
            return '*'
        elif self.value == Truth.NEUTRAL:
            return '?'
        else:
            return '!'

class State(object):
    """
    Represents inference state, with source and target sentences and arbitrary representation.
    """

    def __init__(self, source, target, truth, representation):
        self.source = source
        self.target = target
        self.truth = truth
        self.representation = representation

    def __str__(self):
        return "{} -> {}{}".format(self.source, self.truth.sym(), self.target)

    def __repr__(self):
        return "[S: {}]".format(str(self))

    def isEnd(self):
        """
        @returns: true iff the source sentence is equivalent to the target.
        """
        return self.source == self.target

    @classmethod
    def new(cls, source, target):
        return State(source, target, Truth.TRUE, None)

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

class Agent(ABC):
    """
    A reasoning agent.
    Has parameters to evaluate actions.
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def rerank(self, state, actions):
        """
        @param state: current state
        @param actions: list of valid actions to take
        @returns - an ordered sequence of actions that the agent would take on this state.
        """
        return actions

class Agenda(ABC):
    """
    An agenda handles sorting of states.
    Has parameters to evaluate (state,action)s.
    """

    def __init__(self, parameters, action_generator=lambda _: []):
        self.parameters = parameters
        self.action_generator = action_generator

    def run(self, agent, state):
        if state.isEnd(): return state

        queue = []
        for action in agent.rerank(state, self.action_generator(state)):
            heapq.heappush(queue, (self.score(state, action), (state, action)))

        while len(queue) > 0:
            print(queue)
            _, (state, action) = heapq.heappop(queue)
            # TODO(chaganty): store back-pointers.
            state = action(state)
            if state.isEnd():
                return state

            for action in agent.rerank(state, self.action_generator(state)):
                heapq.heappush(queue, (self.score(state, action), (state, action)))
        return state

    def score(self, state, action):
        """
        score a state to place on the agenda.
        """
        return 0

