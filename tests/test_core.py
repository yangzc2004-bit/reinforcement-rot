"""Unit tests: maze invariants, metrics regressions, memory semantics, client fuse.

Run:  python -m pytest tests/            (or: python -m unittest discover tests)
The mock end-to-end test is the CI gate for the whole L3 engine.
"""
import json
import tempfile
import unittest
from pathlib import Path

from llm.maze import Maze
from llm.memory import SharedMemory
from llm.metrics import population_stats, episode_stats, extract_loops
from llm.client import MockClient, ChatClient, CallLimitExceeded
from llm.agents import SolverAgent


def traj(agent, rnd, path, seed=0, success=False, illegal=0):
    return dict(agent=agent, round=rnd, maze_seed=seed, path=path,
                steps=len(path) - 1, shortest=10, success=success,
                illegal=illegal)


LOOP = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]
PLAIN = [(0, 0), (0, 1)]
OUT = [(0, 1), (0, 0)]


class TestMaze(unittest.TestCase):
    def test_invariants_50_seeds(self):
        for seed in range(50):  # constructor asserts every invariant
            Maze(size=15, delta=4, ring=True, seed=seed)

    def test_delta_must_be_even(self):
        with self.assertRaises(ValueError):
            Maze(size=15, delta=3, seed=0)

    def test_start_not_in_ring(self):
        m = Maze(size=15, delta=4, ring=True, seed=7)
        self.assertNotIn(m.start, m.ring_cells)
        self.assertNotIn(m.goal, m.ring_cells)


class TestMillRate(unittest.TestCase):
    def test_same_agent_two_rounds_is_not_a_mill(self):
        t = [traj(0, 0, PLAIN + LOOP + OUT), traj(0, 1, PLAIN + LOOP + OUT)]
        s = population_stats(t)
        self.assertEqual(s['mill_rate'], 0.0)
        self.assertEqual(s['individual_cycle_rate'], 1.0)

    def test_two_agents_same_round_same_loop_is_a_mill(self):
        t = [traj(0, 0, PLAIN + LOOP + OUT), traj(1, 0, PLAIN + LOOP + OUT)]
        self.assertGreater(population_stats(t)['mill_rate'], 0.0)

    def test_different_loops_are_not_shared(self):
        loop2 = [(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)]
        t = [traj(0, 0, PLAIN + LOOP + OUT),
             traj(1, 0, [(0, 0), (4, 5)] + loop2 + OUT)]
        self.assertEqual(population_stats(t)['mill_rate'], 0.0)

    def test_same_coords_different_maze_not_shared(self):
        t = [traj(0, 0, PLAIN + LOOP + OUT, seed=0),
             traj(1, 0, PLAIN + LOOP + OUT, seed=1)]
        self.assertEqual(population_stats(t)['mill_rate'], 0.0)


class TestLoopErasure(unittest.TestCase):
    def test_figure_eight_yields_loops(self):
        p = [(0, 0), (1, 1), (1, 2), (2, 2), (2, 1), (1, 1),
             (1, 2), (1, 3), (2, 3), (2, 2), (1, 2), (0, 1), (0, 0)]
        loops = extract_loops(p)
        self.assertGreaterEqual(len(loops), 2)

    def test_escape_time(self):
        e = episode_stats(traj(0, 0, PLAIN + LOOP + LOOP + OUT))
        self.assertTrue(e['trapped'])
        self.assertTrue(e['escaped'])
        self.assertIsNotNone(e['escape_time'])


class TestMemory(unittest.TestCase):
    def test_reinforce_keeps_text_history(self):
        m = SharedMemory(lam=1.0)
        m.reinforce((3, 3), 'N', 'first wording', writer=0, round_id=0)
        m.reinforce((3, 3), 'N', 'second wording', writer=1, round_id=1)
        hits = m.retrieve((3, 3))
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]['weight'], 2.0)
        self.assertEqual(hits[0]['text'], 'second wording')  # latest displayed
        self.assertEqual(hits[0]['n_writers'], 2)
        self.assertEqual(len(m.entries[0]['texts']), 2)      # history kept

    def test_evaporation_and_floor(self):
        m = SharedMemory(lam=0.5, floor=0.3)
        m.reinforce((0, 0), 'E', 'x', writer=0, round_id=0)
        m.evaporate()
        self.assertAlmostEqual(m.entries[0]['weight'], 0.5)
        m.evaporate()
        m.evaporate()
        self.assertEqual(len(m.entries), 0)  # fell below floor

    def test_serialization_roundtrip(self):
        m = SharedMemory(lam=0.9)
        m.reinforce((1, 1), 'S', 'hello', writer=0, round_id=0)
        m.reinforce((1, 1), 'S', 'world', writer=1, round_id=1)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'mem.json'
            m.save(p)
            m2 = SharedMemory.load(p, lam=0.9)
        self.assertEqual(m2.entries, m.entries)


class TestClientFuse(unittest.TestCase):
    def test_max_calls_fuse(self):
        c = ChatClient(model='x', api_key='x', base_url='http://localhost:0',
                       max_calls=0)
        with self.assertRaises(CallLimitExceeded):
            c.chat('s', 'u')


class TestMockEndToEnd(unittest.TestCase):
    def test_full_episode_with_memory(self):
        maze = Maze(size=15, delta=4, ring=True, seed=0)
        client = MockClient(seed=0)
        mem = SharedMemory(lam=0.95)
        agent = SolverAgent(client, maze, memory=mem, write_rule='success_only',
                            agent_id=0)
        t = agent.run(max_steps=100, round_id=0, maze_seed=0)
        self.assertIn('path', t)
        self.assertEqual(t['maze_seed'], 0)
        mem.evaporate()
        s = population_stats([t])
        self.assertIn('mill_rate', s)
        self.assertIn('silence', s)


if __name__ == '__main__':
    unittest.main()
