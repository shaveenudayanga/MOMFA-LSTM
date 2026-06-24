import random
import numpy as np


class ParetoArchive:
    """Non-dominated archive for the MOMFA bi-objective search.

    Objectives (both minimised):
        f1 = val RMSE      (prediction accuracy)
        f2 = param count   (model complexity)

    Per D-002: each entry stores the PHENOTYPE (fixed binary mask + decoded
    hyperparameters + fitness values) plus the continuous position that produced
    it. The position is stored only to serve as x_j in the FA attraction formula;
    it plays no role in dominance comparison.
    """

    def __init__(self):
        self._entries: list[dict] = []

    # ── Dominance ────────────────────────────────────────────────────────────

    @staticmethod
    def _dominates(a: dict, b: dict) -> bool:
        """True if solution a Pareto-dominates solution b.

        a dominates b iff a is no worse on every objective AND strictly
        better on at least one.
        """
        return (
            a['rmse']     <= b['rmse']     and
            a['n_params'] <= b['n_params'] and
            (a['rmse'] < b['rmse'] or a['n_params'] < b['n_params'])
        )

    # ── Mutation ─────────────────────────────────────────────────────────────

    def update(self, candidate: dict) -> bool:
        """Attempt to add candidate to the archive.

        candidate must have keys: mask, hyperparams, position, rmse, n_params.

        Steps:
          1. If any existing entry dominates candidate → discard, return False.
          2. Remove all existing entries dominated by candidate.
          3. Add candidate, return True.
        """
        for entry in self._entries:
            if self._dominates(entry, candidate):
                return False

        self._entries = [
            e for e in self._entries if not self._dominates(candidate, e)
        ]
        self._entries.append(candidate)
        return True

    # ── Access ───────────────────────────────────────────────────────────────

    def select_guide(self) -> dict:
        """Return a uniformly random archive member.

        Per CONTEXT.md §3.4: 'Fireflies move toward a randomly selected archive
        member (not single global best).' All archive members are non-dominated
        so no ranking is needed.

        Raises RuntimeError if the archive is empty (should not happen after
        the initial population evaluation).
        """
        if not self._entries:
            raise RuntimeError("Archive is empty - cannot select a guide.")
        return random.choice(self._entries)

    def get_front(self) -> list[dict]:
        """Return all non-dominated solutions (the current Pareto front)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ParetoArchive({len(self._entries)} solutions)"
