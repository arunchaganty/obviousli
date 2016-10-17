#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility routines
"""

def edit_distance(l1, l2):
    """
    Computes edit distance between two sequences, l1, l2
    """
    if len(l1) < len(l2):
        return edit_distance(l2, l1)

    # len(l1) >= len(l2)
    if len(l2) == 0:
        return len(l1)

    previous_row = list(range(len(l2) + 1))
    for i, c1 in enumerate(l1):
        current_row = [i + 1]
        for j, c2 in enumerate(l2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than l2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def normalized_edit_distance(l1, l2):
    """
    @returns: `edit_distance` normalized by length of input (lies between 0, 1).
    """
    return edit_distance(l1, l2) / max(len(l1), len(l2))
