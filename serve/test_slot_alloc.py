"""serve/test_slot_alloc.py - which batch slot a request gets (StrataEngine.pick_slot), no engine needed.
strata-poly runs its slots as one group (the Qwen engine's pipelined --batch-groups are gone, phase 1d).

    python -m unittest serve.test_slot_alloc -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine  # noqa: E402


def engine(batch):
    e = object.__new__(StrataEngine)          # pick_slot reads only the slot tables
    e.batch = batch
    e.slot_order = list(range(batch))
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [0.0] * batch
    return e


class OneGroup(unittest.TestCase):
    def test_reuse_then_least_recently_used(self):
        e = engine(4)
        e.slot_held[2] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # its conversation's next turn
        e.slot_held[2] = [9]
        e.slot_held = [[5], [], [9], [6]]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # an empty slot before evicting a held one
        e.slot_held[1] = [8]
        e.slot_used = [5.0, 3.0, 9.0, 1.0]
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)        # else the least recently used


class TheSlotOfTheChat(unittest.TestCase):
    """Each slot keeps its chat (#2518): the next turn goes back to it."""

    def test_the_slot_of_the_chat(self):
        T = StrataEngine.TURN_ID
        e = engine(2)
        hist_a, hist_b = list(range(1000, 1700)), list(range(2000, 2700))
        e.slot_held = [hist_a + [T, 7, 8] + [50, 51], hist_b + [T, 7, 8] + [60]]
        e.slot_used = [10.0, 20.0]
        again = lambda h: h + [T, 5, 6] + list(range(3000, 3050))
        self.assertEqual(e.pick_slot(again(hist_b)), 1)
        self.assertEqual(e.pick_slot(again(hist_a)), 0)
        self.assertEqual(e.pick_slot(list(range(5000, 5100))), 0)      # a new chat: the least recently used one


if __name__ == "__main__":
    unittest.main()
