"""Solver agent: walks the maze one cell per LLM call, optionally reading and
writing the shared memory.

Prompt contract (kept minimal so results are comparable across models):
  system: role + rules + the single-letter MOVE: X output format
  user  : current cell, open directions, own visited marks (if breadcrumbs),
          top-k shared-memory entries at this cell (if memory enabled)

Write rules (B5 ablation axis):
  none                   : never write (pure individual run)
  universal              : write advice at every visited cell
  success_only           : write only if the run reached the goal
  efficient_success_only : write only if steps <= eff_factor * shortest path
"""
import re

SYSTEM = """You are a maze-solving agent. You move one cell per turn on a grid.
Directions: N (up), S (down), E (right), W (left). You only see your current
cell, its open directions, and notes left by other agents (if any).
Reach the goal G as fast as possible. Reply with exactly one line:
MOVE: X
where X is one of the open directions."""


def parse_move(text, legal):
    m = re.search(r'MOVE:\s*([NSEW])', text.upper())
    if m and m.group(1) in legal:
        return m.group(1), True
    for ch in text.upper():
        if ch in legal:
            return ch, False  # recovered but malformed -> counts as non-silent violation
    return None, False


class SolverAgent:
    def __init__(self, client, maze, memory=None, breadcrumbs=True,
                 write_rule='success_only', eff_factor=1.25, agent_id=0):
        self.client = client
        self.maze = maze
        self.memory = memory
        self.breadcrumbs = breadcrumbs
        self.write_rule = write_rule
        self.eff_factor = eff_factor
        self.id = agent_id

    def _prompt(self, cell, visited, round_id):
        lines = [f"CELL: {cell[0]},{cell[1]}",
                 f"OPEN: {''.join(self.maze.open_dirs(cell))}"]
        if self.breadcrumbs and visited:
            lines.append("VISITED: " + ' '.join(f"{r},{c}" for r, c in sorted(visited)))
        if self.memory is not None:
            hits = self.memory.retrieve(cell)
            if hits:
                lines.append("NOTES from other agents at this cell:")
                for h in hits:
                    lines.append(f"- go {h['direction']} (weight {h['weight']:.2f}): {h['text']}")
        lines.append("Your move?")
        return '\n'.join(lines)

    def run(self, max_steps=200, round_id=0):
        """One episode. Returns a trajectory dict."""
        cell = self.maze.start
        path = [cell]
        visited = {cell}
        illegal = 0
        steps = 0
        while cell != self.maze.goal and steps < max_steps:
            legal = self.maze.open_dirs(cell)
            here = cell
            resp = self.client.chat(SYSTEM, self._prompt(cell, visited, round_id))
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
            if self.memory is not None and self.write_rule == 'universal':
                self.memory.reinforce(here, d, "agent passed through",
                                      self.id, round_id)
        success = cell == self.maze.goal
        shortest = self.maze.shortest_path_len()
        if self.memory is not None and success and self.write_rule in (
                'success_only', 'efficient_success_only'):
            efficient = steps <= self.eff_factor * shortest
            if self.write_rule == 'success_only' or efficient:
                for i in range(len(path) - 1):
                    c, n = path[i], path[i + 1]
                    for d in self.maze.open_dirs(c):
                        if self.maze.move(c, d) == n:
                            self.memory.reinforce(c, d, "on a successful route",
                                                  self.id, round_id)
                            break
        return dict(agent=self.id, round=round_id, path=path, steps=steps,
                    success=success, illegal=illegal, shortest=shortest)
