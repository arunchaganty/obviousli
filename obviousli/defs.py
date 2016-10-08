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
        return "[S: {} -> {}{}]".format(self.source, self.truth, self.target)

    def isEnd(self):
        """
        @returns: true iff the source sentence is equivalent to the target.
        """
        return self.source == self.target

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

class Agent(ABC):
    """
    A reasoning agent.
    Has parameters to evaluate actions.
    """

    def __init__(self, parameters):
        self.parameters = parameters

    @abstractmethod
    def act(self, state):
        """
        TODO(chaganty): allow returning multiple best actions
        @returns - the action an agent would take on this state.
        """
        pass

class Agenda(ABC):
    """
    An agenda handles sorting of states.
    Has parameters to evaluate states.
    """

    def __init__(self, parameters):
        self.parameters = parameters

    def run(self, agent, state):
        queue = [(0,state)]

        while len(queue) > 0:
            _, state = heapq.heappop(queue)
            # TODO(chaganty): store back-pointers.
            state_ = agent.act(state)(state)
            if state_.isEnd():
                return state_

            heapq.heappush((self.score(state_), state_))
        return state

    def score(self, state):
        """
        score a state to place on the agenda.
        """
        return 0

