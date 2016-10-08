#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from .defs import Agent
from .actions import GiveUpAction

class GiveUpAgent(Agent):
    def __init__(self):
        super(GiveUpAgent, self).__init__([])

    def act(self, state):
        return GiveUpAction()

