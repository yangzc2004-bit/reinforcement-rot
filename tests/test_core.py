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
from llm.agents import SYSTEM, SolverAgent


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

    def test_small_calibration_maze_keeps_controlled_cycles(self):
        maze = Maze(size=6, delta=2, ring=True, seed=0,
                    min_shortest=10, max_backbone=10, neck_min_dist=2)
        self.assertEqual(maze.shortest_path_len(), 10)

    def test_start_not_in_ring(self):
        m = Maze(size=15, delta=4, ring=True, seed=7)
        self.assertNotIn(m.start, m.ring_cells)
        self.assertNotIn(m.goal, m.ring_cells)

    def test_each_fill_component_has_one_controlled_root(self):
        m = Maze(size=10, delta=2, ring=True, seed=7,
                 min_shortest=18, max_backbone=18, neck_min_dist=5)
        self.assertEqual(len(m.fill_components), len(m.fill_roots))
        for component, root in zip(m.fill_components, m.fill_roots):
            cells = set(component)
            external_edges = [
                (cell, m.move(cell, direction))
                for cell in cells
                for direction in m.open_dirs(cell)
                if m.move(cell, direction) not in cells
            ]
            self.assertEqual(len(external_edges), 1)
            self.assertEqual(frozenset(external_edges[0]),
                             frozenset((root['root'], root['anchor'])))


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

    def test_sustained_two_cell_oscillation_is_separate_from_ring_loop(self):
        p = [(0, 0), (0, 1), (0, 0), (0, 1), (0, 0)]
        e = episode_stats(traj(0, 0, p))
        self.assertEqual(e['cycle_rate'], 0.0)
        self.assertEqual(e['oscillation_rate'], 1.0)
        self.assertEqual(e['longest_oscillation'], 4)

    def test_single_backtrack_is_not_sustained_oscillation(self):
        p = [(0, 0), (0, 1), (0, 0), (1, 0)]
        e = episode_stats(traj(0, 0, p))
        self.assertEqual(e['oscillation_rate'], 0.0)
        self.assertEqual(e['longest_oscillation'], 0)


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

    def test_mechanical_writes_are_counted_separately(self):
        class LineMaze:
            start = (0, 0)
            goal = (0, 2)

            def open_dirs(self, cell):
                return {
                    (0, 0): ['E'],
                    (0, 1): ['E', 'W'],
                    (0, 2): ['W'],
                }[cell]

            def move(self, cell, direction):
                delta = {'E': (0, 1), 'W': (0, -1)}[direction]
                return cell[0] + delta[0], cell[1] + delta[1]

            def shortest_path_len(self):
                return 2

        class EastClient:
            def chat(self, system, user, max_tokens=None):
                return 'MOVE: E'

        memory = SharedMemory(lam=1.0)
        result = SolverAgent(
            EastClient(), LineMaze(), memory=memory,
            write_rule='success_only',
        ).run(max_steps=5)
        self.assertTrue(result['success'])
        self.assertEqual(result['memory_writes'], 2)
        self.assertEqual(result['mechanical_writes'], 2)
        self.assertEqual(result['authored_writes'], 0)
        self.assertEqual(result['notes_written'], 0)


class TestTrapConsecutiveRuns(unittest.TestCase):
    """A trap is a CONSECUTIVE repeat of one canonical loop, not a total count:
    interleaved A,B,A is exploration; A,A is a mill."""

    def test_interleaved_loops_are_not_a_trap(self):
        loop2 = [(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)]
        p = PLAIN + LOOP + [(2, 2), (4, 5)] + loop2 + [(5, 5), (2, 2)] + LOOP + OUT
        e = episode_stats(traj(0, 0, p))
        self.assertFalse(e['trapped'])
        self.assertIsNone(e['escape_time'])

    def test_consecutive_repeat_is_a_trap(self):
        e = episode_stats(traj(0, 0, PLAIN + LOOP + LOOP + OUT))
        self.assertTrue(e['trapped'])
        self.assertTrue(e['escaped'])
        self.assertIsNotNone(e['escape_time'])

    def test_consecutive_repeat_to_death(self):
        e = episode_stats(traj(0, 0, PLAIN + LOOP + LOOP))
        self.assertTrue(e['trapped'])
        self.assertFalse(e['escaped'])


class TestJsonlCorruptTail(unittest.TestCase):
    def _write(self, path, lines, final_newline=True):
        text = '\n'.join(lines)
        if final_newline:
            text += '\n'
        path.write_text(text)

    def test_truncated_last_line_is_dropped(self):
        import llm.run_b2 as b2
        good = json.dumps(dict(maze_seed=0, round_retention=1.0, round=0,
                               agent=0, traj={}))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.jsonl'
            self._write(p, [good, '{"maze_seed": 0, "round_re'],
                        final_newline=False)
            old = b2.EPISODES_LOG
            b2.EPISODES_LOG = p
            try:
                done = b2._load_done()
            finally:
                b2.EPISODES_LOG = old
        self.assertEqual(len(done), 1)  # good line kept, tail dropped

    def test_truncated_tail_is_repaired_before_append(self):
        import llm.run_b2 as b2
        good0 = json.dumps(dict(maze_seed=0, round_retention=1.0, round=0,
                                agent=0, traj={}))
        good1 = json.dumps(dict(maze_seed=0, round_retention=1.0, round=0,
                                agent=1, traj={}))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.jsonl'
            self._write(p, [good0, '{"truncated'], final_newline=False)
            old = b2.EPISODES_LOG
            b2.EPISODES_LOG = p
            try:
                b2._load_done()
                with p.open('a', encoding='utf-8') as f:
                    f.write(good1 + '\n')
                done = b2._load_done()
            finally:
                b2.EPISODES_LOG = old
        self.assertEqual(len(done), 2)  # new record is readable after repair

    def test_newline_terminated_invalid_tail_is_not_silently_dropped(self):
        import llm.run_b2 as b2
        good = json.dumps(dict(maze_seed=0, round_retention=1.0, round=0,
                               agent=0, traj={}))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.jsonl'
            self._write(p, [good, '{"broken"'])
            p.write_text(p.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            old = b2.EPISODES_LOG
            b2.EPISODES_LOG = p
            try:
                with self.assertRaises(ValueError):
                    b2._load_done()
            finally:
                b2.EPISODES_LOG = old

    def test_corrupt_middle_line_raises(self):
        import llm.run_b2 as b2
        good = json.dumps(dict(maze_seed=0, round_retention=1.0, round=0,
                               agent=0, traj={}))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.jsonl'
            self._write(p, [good, '{"broken"', good])
            old = b2.EPISODES_LOG
            b2.EPISODES_LOG = p
            try:
                with self.assertRaises(ValueError):
                    b2._load_done()
            finally:
                b2.EPISODES_LOG = old


class TestB2Conditions(unittest.TestCase):
    def test_no_memory_and_shared_records_have_distinct_checkpoint_keys(self):
        import llm.run_b2 as b2

        base = dict(maze_seed=7, round_retention=None, round=0,
                    agent=0, traj={})
        no_memory = dict(base, memory_condition='none')
        shared = dict(base, memory_condition='shared')
        self.assertNotEqual(b2._episode_key(no_memory),
                            b2._episode_key(shared))

    def test_no_memory_control_is_enabled_by_default(self):
        import llm.run_b2 as b2

        conditions = b2._conditions(dict(round_retentions=[1.0, 0.85]))
        self.assertEqual(conditions[0], {
            'memory_condition': 'none',
            'round_retention': None,
        })
        self.assertEqual(len(conditions), 3)


class TestClientFuse(unittest.TestCase):
    def test_deepseek_thinking_is_disabled(self):
        from unittest import mock

        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    'choices': [{'message': {'content': 'MOVE: N'}}],
                    'usage': {'total_tokens': 3},
                }).encode()

        def fake_urlopen(req, timeout):
            captured['body'] = json.loads(req.data.decode())
            return Response()

        c = ChatClient(model='deepseek-v4-flash', api_key='x',
                       base_url='http://example.test', thinking=False)
        with mock.patch('urllib.request.urlopen', side_effect=fake_urlopen):
            c.chat('system', 'user')
        self.assertEqual(captured['body']['thinking'], {'type': 'disabled'})

    def test_non_deepseek_provider_does_not_receive_thinking_field(self):
        from unittest import mock

        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    'choices': [{'message': {'content': 'MOVE: N'}}],
                    'usage': {},
                }).encode()

        def fake_urlopen(req, timeout):
            captured['body'] = json.loads(req.data.decode())
            return Response()

        c = ChatClient(model='some-openai-compatible-model', api_key='x',
                       base_url='http://example.test')
        with mock.patch('urllib.request.urlopen', side_effect=fake_urlopen):
            c.chat('system', 'user')
        self.assertNotIn('thinking', captured['body'])

    def test_max_calls_fuse(self):
        c = ChatClient(model='x', api_key='x', base_url='http://localhost:0',
                       max_calls=0)
        with self.assertRaises(CallLimitExceeded):
            c.chat('s', 'u')

    def test_fuse_counts_attempts_not_successes(self):
        # retries (failed attempts) burn the budget too — they cost money
        from unittest import mock
        c = ChatClient(model='x', api_key='x', base_url='http://localhost:0',
                       max_calls=3)
        self.assertEqual(c.n_attempts, 0)
        self.assertEqual(c.n_calls, 0)
        with mock.patch('time.sleep'):  # skip real backoff waits
            with self.assertRaises(CallLimitExceeded):
                c.chat('s', 'u')  # all attempts fail (nothing on port 0)
        self.assertEqual(c.n_attempts, 3)   # fuse tripped at attempt #3
        self.assertEqual(c.n_calls, 0)      # no successes
        self.assertLessEqual(c.n_attempts, c.max_calls)


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


class TestNavigationLedger(unittest.TestCase):
    def test_prompt_exposes_tried_untried_and_backtrack(self):
        maze = Maze(size=15, delta=4, ring=True, seed=0)
        agent = SolverAgent(MockClient(seed=0), maze, memory=None,
                            navigation_ledger=True)
        cell = (1, 0)
        prompt = agent._prompt(
            cell,
            visited={(0, 0), cell},
            attempted={((1, 0), 'N')},
            parents={cell: (0, 0)},
            path=[(0, 0), cell],
        )
        self.assertIn('TRIED_OPEN: N', prompt)
        self.assertIn('UNTRIED_OPEN: E', prompt)
        self.assertIn('BACKTRACK: N', prompt)
        self.assertIn('RECENT_PATH: 0,0 -> 1,0', prompt)
        self.assertIn('NAV_RULE:', prompt)

    def test_visit_only_prompt_omits_navigation_ledger(self):
        maze = Maze(size=15, delta=4, ring=True, seed=0)
        agent = SolverAgent(MockClient(seed=0), maze, memory=None,
                            breadcrumbs=True, navigation_ledger=False,
                            navigation_guard=False)
        prompt = agent._prompt(
            (1, 0),
            visited={(0, 0), (1, 0)},
            attempted={((1, 0), 'N')},
            parents={(1, 0): (0, 0)},
            path=[(0, 0), (1, 0)],
        )
        self.assertIn('VISITED:', prompt)
        self.assertIn('0,0 -> 1,0', prompt)
        self.assertIn('NEIGHBORS:', prompt)
        self.assertNotIn('TRIED_OPEN:', prompt)
        self.assertNotIn('UNTRIED_OPEN:', prompt)
        self.assertNotIn('BACKTRACK:', prompt)
        self.assertNotIn('NAV_RULE:', prompt)
        self.assertIn('loop-erase VISITED', SYSTEM)
        self.assertNotIn('reversing your latest move', SYSTEM)

    def test_visit_only_prompt_preserves_chronological_revisits(self):
        maze = Maze(size=15, delta=4, ring=True, seed=0)
        agent = SolverAgent(MockClient(seed=0), maze, memory=None,
                            breadcrumbs=True, navigation_ledger=False,
                            navigation_guard=False)
        prompt = agent._prompt(
            (1, 0),
            visited={(0, 0), (1, 0)},
            path=[(0, 0), (1, 0), (0, 0), (1, 0)],
        )
        self.assertIn(
            'VISITED: (oldest->newest) 0,0 -> 1,0 -> 0,0 -> 1,0',
            prompt,
        )
        self.assertNotIn('RECENT_PATH:', prompt)

    def test_pilot_stagnation_detector_does_not_change_moves(self):
        class AlwaysSouth:
            n_calls = 0
            n_attempts = 0
            n_tokens = 0
            n_errors = 0

            def chat(self, system, user, max_tokens=None):
                self.n_calls += 1
                self.n_attempts += 1
                return 'MOVE: S'

        maze = Maze(size=15, delta=4, ring=True, seed=0)
        result = SolverAgent(
            AlwaysSouth(), maze, memory=None, stagnation_repeats=3
        ).run(max_steps=300, maze_seed=0)
        self.assertTrue(result['stagnation_abort'])
        self.assertLess(result['steps'], 300)
        self.assertEqual(result['navigation_overrides'], 0)

    def test_navigation_guard_avoids_visited_cells(self):
        maze = Maze(size=15, delta=4, ring=True, seed=0)
        client = MockClient(seed=0)
        agent = SolverAgent(client, maze, memory=None,
                            navigation_guard=True)
        result = agent.run(max_steps=300, maze_seed=0)
        self.assertTrue(result['success'])
        self.assertGreater(result['navigation_overrides'], 0)


if __name__ == '__main__':
    unittest.main()
