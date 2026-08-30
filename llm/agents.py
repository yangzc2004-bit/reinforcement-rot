"""Solver agent: walks the maze one cell per LLM call, optionally reading and
writing the shared memory.

Prompt contract (kept minimal so results are comparable across models):
  system: role + rules + the single-letter MOVE: X output format
  user  : current cell, GOAL coordinates, open directions, own visited marks
          (if breadcrumbs), top-k shared-memory entries at this cell
          (if memory enabled)

GOAL coordinates are given (the L3 analog of L2's outbound merit bias): without
them there is no history-vs-geometry tension and B1 measures exploration luck,
not capability. The full map is NEVER provided.

Write rules (B5 ablation axis — {mechanical vs authored} x {success-gated vs all}):
  none                   : never write (pure individual run)
  success_only           : mechanical path distill, only if the run succeeded
  efficient_success_only : mechanical, only if steps <= eff_factor * shortest
  agent_authored         : on success, the agent REVIEWS its own path and writes
                           natural-language notes for the junctions it chose at
  universal              : succeed or fail, the agent writes notes for every
                           junction it visited (dead ends included, as warnings)

Authored notes are validated before entering memory: the cell must lie on the
agent's path and the advised direction must be open there. Malformed notes are
dropped (counted in n_notes_dropped).
"""
import re

SYSTEM = """You are a maze-solving agent. You move one cell per turn on a grid.
Directions: N (up), S (down), E (right), W (left). You only see your current
cell, its open directions, and notes left by other agents (if any).
Reach the goal G as fast as possible. Reply with exactly one line:
MOVE: X
where X is one of the open directions."""

AUTHOR_SYSTEM = """You just walked a maze from the start to the goal (or died trying).
You are now writing notes into a SHARED memory that later agents will read at
the exact cell they apply to. Write short, concrete, honest advice: which way
to go at a junction, which branch loops back or dead-ends. You may be wrong;
later agents will judge. You know the goal's coordinates but never saw a map;
do not invent distances or claim knowledge of cells you did not visit. Reply
with one note per line, EXACTLY in this format:
NOTE: r,c | D | your advice
where (r,c) is a junction listed below and D is one of its open directions.
Write at most {max_notes} notes. If nothing is worth saying, reply: NONE"""


def parse_move(text, legal):
    m = re.search(r'MOVE:\s*([NSEW])', text.upper())
    if m and m.group(1) in legal:
        return m.group(1), True
    for ch in text.upper():
        if ch in legal:
            return ch, False  # recovered but malformed -> counts as non-silent violation
    return None, False


NOTE_RE = re.compile(r'NOTE:\s*(\d+)\s*,\s*(\d+)\s*\|\s*([NSEW])\s*\|\s*(.+)')


class SolverAgent:
    def __init__(self, client, maze, memory=None, breadcrumbs=True,
                 write_rule='success_only', eff_factor=1.25, agent_id=0,
                 max_notes=8):
        self.client = client
        self.maze = maze
        self.memory = memory
        self.breadcrumbs = breadcrumbs
        self.write_rule = write_rule
        self.eff_factor = eff_factor
        self.id = agent_id
        self.max_notes = max_notes
        self.n_notes_written = 0
        self.n_notes_dropped = 0

    def _prompt(self, cell, visited):
        # GOAL is given as coordinates: the L3 analog of L2's outbound merit
        # bias toward the food source. Without it the history-vs-geometry
        # tension collapses (no geometry leg), and B1 measures exploration
        # luck instead of capability. Full map is NEVER provided.
        lines = [f"CELL: {cell[0]},{cell[1]}",
                 f"GOAL: {self.maze.goal[0]},{self.maze.goal[1]}",
                 f"OPEN: {''.join(self.maze.open_dirs(cell))}"]
        if self.breadcrumbs and visited:
            lines.append("VISITED: " + ' '.join(f"{r},{c}" for r, c in sorted(visited)))
        if self.memory is not None:
            hits = self.memory.retrieve(cell)
            if hits:
                lines.append("NOTES from other agents at this cell:")
                for h in hits:
                    lines.append(f"- go {h['direction']} (weight {h['weight']:.2f}, "
                                 f"{h['n_writers']} agent(s)): {h['text']}")
        lines.append("Your move?")
        return '\n'.join(lines)

    def _junctions(self, path):
        """Decision points on the path: visited cells with a real choice."""
        on_path = set(path)
        jcts = {}
        for c in on_path:
            o = self.maze.open_dirs(c)
            if len(o) >= 3:
                jcts[c] = o
        return jcts

    def _author_notes(self, path, steps, shortest, success, round_id):
        jcts = self._junctions(path)
        if not jcts:
            return
        lines = [f"Outcome: {'reached the goal' if success else 'ran out of steps'}.",
                 f"Steps taken: {steps}. (The true shortest path is {shortest} steps.)",
                 "Junctions you visited (cell -> open directions):"]
        for (r, c), o in sorted(jcts.items()):
            lines.append(f"JCT {r},{c} OPEN={''.join(o)}")
        lines.append("Write your notes now.")
        resp = self.client.chat(
            AUTHOR_SYSTEM.replace('{max_notes}', str(self.max_notes)),
            '\n'.join(lines), max_tokens=80 * self.max_notes)
        for m in NOTE_RE.finditer(resp):
            cell = (int(m.group(1)), int(m.group(2)))
            d, text = m.group(3), m.group(4).strip()[:200]
            if cell in jcts and d in jcts[cell]:
                self.memory.reinforce(cell, d, text, self.id, round_id)
                self.n_notes_written += 1
            else:
                self.n_notes_dropped += 1

    def run(self, max_steps=200, round_id=0, maze_seed=None):
        """One episode. Returns a trajectory dict (maze_seed is recorded so that
        metrics can group loops by maze — identical coordinates in different
        mazes are NOT the same loop)."""
        cell = self.maze.start
        path = [cell]
        visited = {cell}
        illegal = 0
        steps = 0
        while cell != self.maze.goal and steps < max_steps:
            legal = self.maze.open_dirs(cell)
            resp = self.client.chat(SYSTEM, self._prompt(cell, visited))
            d, ok = parse_move(resp, legal)
            if not ok:
                illegal += 1
            if d is None:
                d = legal[0]
            nxt = self.maze.move(cell, d)
            if nxt is None:  # wall (should not happen after parse); count illegal
                illegal += 1
                nxt = cell
            cell = nxt
            path.append(cell)
            visited.add(cell)
            steps += 1
        success = cell == self.maze.goal
        shortest = self.maze.shortest_path_len()

        if self.memory is not None:
            if self.write_rule == 'agent_authored' and success:
                self._author_notes(path, steps, shortest, True, round_id)
            elif self.write_rule == 'universal':
                self._author_notes(path, steps, shortest, success, round_id)
            elif success and self.write_rule in ('success_only', 'efficient_success_only'):
                efficient = steps <= self.eff_factor * shortest
                if self.write_rule == 'success_only' or efficient:
                    for i in range(len(path) - 1):
                        c, n = path[i], path[i + 1]
                        for d in self.maze.open_dirs(c):
                            if self.maze.move(c, d) == n:
                                self.memory.reinforce(c, d, "on a successful route",
                                                      self.id, round_id)
                                break
        return dict(agent=self.id, round=round_id, maze_seed=maze_seed,
                    path=path, steps=steps,
                    success=success, illegal=illegal, shortest=shortest,
                    notes_written=self.n_notes_written,
                    notes_dropped=self.n_notes_dropped)
