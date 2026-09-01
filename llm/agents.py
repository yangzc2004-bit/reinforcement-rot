"""Solver agent: walks the maze one cell per LLM call, optionally reading and
writing the shared memory.

Prompt contract (kept minimal so results are comparable across models):
  system: role + rules + the single-letter MOVE: X output format
    user  : current cell, GOAL coordinates, open directions, structured private
          exploration ledger, top-k shared-memory entries at this cell
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
Directions: N=(r-1,c), S=(r+1,c), E=(r,c+1), W=(r,c-1). You only see your
current cell, its open directions, and notes left by other agents (if any).
VISITED is your private chronological trail from oldest to newest; its final
cell is your current location. For every OPEN direction, calculate its
destination coordinate. If any destination has never appeared in VISITED,
choose an unvisited destination (prefer one closer to GOAL). Only when every
open destination is visited should you backtrack. To find the backtrack cell,
loop-erase VISITED: scan it oldest to newest, and whenever a cell repeats,
delete everything after its previous occurrence before continuing. The
resulting route ends at CELL; move to the cell immediately before CELL on that
route. This means that after returning from a dead end you continue toward an
earlier junction instead of alternating between the same two cells.
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
                 max_notes=8, navigation_ledger=False,
                 navigation_guard=False, stagnation_repeats=0):
        self.client = client
        self.maze = maze
        self.memory = memory
        self.breadcrumbs = breadcrumbs
        self.write_rule = write_rule
        self.eff_factor = eff_factor
        self.id = agent_id
        self.max_notes = max_notes
        self.navigation_ledger = navigation_ledger
        self.navigation_guard = navigation_guard
        self.stagnation_repeats = max(0, int(stagnation_repeats))
        self.n_notes_written = 0
        self.n_notes_dropped = 0
        self.n_memory_writes = 0
        self.n_mechanical_writes = 0

    def _prompt(self, cell, visited, attempted=None, parents=None, path=None):
        # GOAL is given as coordinates: the L3 analog of L2's outbound merit
        # bias toward the food source. Without it the history-vs-geometry
        # tension collapses (no geometry leg), and B1 measures exploration
        # luck instead of capability. Full map is NEVER provided.
        attempted = attempted or set()
        parents = parents or {}
        path = path or []
        lines = [f"CELL: {cell[0]},{cell[1]}",
                 f"GOAL: {self.maze.goal[0]},{self.maze.goal[1]}",
                 f"OPEN: {''.join(self.maze.open_dirs(cell))}"]
        neighbors = []
        for direction in self.maze.open_dirs(cell):
            nxt = self.maze.move(cell, direction)
            neighbors.append(f"{direction}=({nxt[0]},{nxt[1]})")
        if neighbors:
            lines.append("NEIGHBORS: " + ' '.join(neighbors))
        if self.breadcrumbs and path:
            lines.append(
                "VISITED: (oldest->newest) "
                + " -> ".join(f"{r},{c}" for r, c in path))
        if self.navigation_ledger:
            legal = self.maze.open_dirs(cell)
            tried = [d for d in legal if (cell, d) in attempted]
            untried = [d for d in legal
                       if (cell, d) not in attempted
                       and self.maze.move(cell, d) not in visited]
            lines.append("TRIED_OPEN: " + (''.join(tried) or "NONE"))
            lines.append("UNTRIED_OPEN: " + (''.join(untried) or "NONE"))
            parent = parents.get(cell)
            backtrack = None
            if parent is not None:
                backtrack = self.maze._dir_between(cell, parent)
            lines.append("BACKTRACK: " + (backtrack or "NONE"))
            if path:
                recent = " -> ".join(f"{r},{c}" for r, c in path[-10:])
                lines.append("RECENT_PATH: " + recent)
            lines.append(
                "NAV_RULE: choose UNTRIED_OPEN first; if it is NONE, choose "
                "BACKTRACK. Do not reuse a TRIED_OPEN edge unless it is "
                "BACKTRACK.")
        if self.navigation_guard:
            lines.append(
                "NAV_GUARD: the environment rejects moves to visited cells "
                "while UNTRIED_OPEN exists.")
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
                self.n_memory_writes += 1
            else:
                self.n_notes_dropped += 1

    def run(self, max_steps=200, round_id=0, maze_seed=None):
        """One episode. Returns a trajectory dict (maze_seed is recorded so that
        metrics can group loops by maze — identical coordinates in different
        mazes are NOT the same loop)."""
        self.n_notes_written = 0
        self.n_notes_dropped = 0
        self.n_memory_writes = 0
        self.n_mechanical_writes = 0
        cell = self.maze.start
        path = [cell]
        visited = {cell}
        attempted = set()
        parents = {}
        navigation_overrides = 0
        state_visits = {}
        stagnation_abort = False
        illegal = 0
        steps = 0
        while cell != self.maze.goal and steps < max_steps:
            state = (cell, frozenset(visited))
            state_visits[state] = state_visits.get(state, 0) + 1
            if (self.stagnation_repeats
                    and state_visits[state] > self.stagnation_repeats):
                stagnation_abort = True
                break
            legal = self.maze.open_dirs(cell)
            resp = self.client.chat(
                SYSTEM, self._prompt(cell, visited, attempted, parents, path))
            d, ok = parse_move(resp, legal)
            if not ok:
                illegal += 1
            if d is None:
                d = legal[0]
            if self.navigation_guard:
                unvisited = [direction for direction in legal
                             if self.maze.move(cell, direction) not in visited]
                parent = parents.get(cell)
                backtrack = (self.maze._dir_between(cell, parent)
                             if parent is not None else None)
                if unvisited:
                    if d not in unvisited:
                        d = unvisited[0]
                        navigation_overrides += 1
                elif backtrack in legal:
                    if d != backtrack:
                        navigation_overrides += 1
                    d = backtrack
                else:
                    break
            attempted.add((cell, d))
            nxt = self.maze.move(cell, d)
            if nxt is None:  # wall (should not happen after parse); count illegal
                illegal += 1
                nxt = cell
            elif nxt not in visited:
                parents[nxt] = cell
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
                                self.n_memory_writes += 1
                                self.n_mechanical_writes += 1
                                break
        return dict(agent=self.id, round=round_id, maze_seed=maze_seed,
                    path=path, steps=steps,
                    success=success, illegal=illegal, shortest=shortest,
                    navigation_overrides=navigation_overrides,
                    stagnation_abort=stagnation_abort,
                    notes_written=self.n_notes_written,
                    notes_dropped=self.n_notes_dropped,
                    memory_writes=self.n_memory_writes,
                    mechanical_writes=self.n_mechanical_writes,
                    authored_writes=self.n_notes_written)
