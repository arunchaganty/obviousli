#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agendas
"""

from .defs import Agenda
from .util import edit_distance

class EditDistanceAgenda(Agenda):
    def score(self, state, action):
        return edit_distance(state.source, state.target)

