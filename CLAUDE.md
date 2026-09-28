# MOMFA-LSTM research implementation

This repo implements a final-year research project: a Multi-Objective Modified
Firefly Algorithm (MOMFA) that jointly selects technical indicators and tunes
LSTM hyperparameters for next-day close price forecasting on the Colombo Stock
Exchange.

## The specification

`MOMFA_research_proposal.txt` (repo root) is the single source of truth.

- Sections 0-8: the approved research proposal (WHAT the research is).
- Sections 9-12: implementation addendum (data split, decoding, compute budget,
  results logging, figures, sanity checks).
- Section 13: working rules you MUST follow.

Before writing or changing any code, read the relevant sections of that file.
At the start of a new session, read `docs/STATUS.md` first (current state, open items,
how to run things), then Section 0 (quick reference) and Section 13 of the spec.

Notes marked `[DECIDED date]` in the spec are author-approved changes; the reasoning is in
`docs/DECISIONS.md`. If documents disagree: spec (with its [DECIDED] notes) >
`docs/DECISIONS.md` > everything else. Fix a disagreement when you find it.

Project docs (all in `docs/`): STATUS.md (state, update at session end), DECISIONS.md (why),
FINDINGS.md (bugs, results, open questions, with numbers), RESULTS_PLAN.md (figures and
tables), CONTEXT.md (short summary, not authoritative).

## Non-negotiable rules

1. Do not invent methods, parameters or shortcuts that are not in the spec. If
   something is missing, ambiguous or marked [OPEN], stop and ask me. Do not
   choose silently.
2. Cite the spec section your code implements (e.g. "3.E.4, 9.6").
3. Log every deviation from the spec in `docs/DECISIONS.md` (date, section, what,
   why) and tell me about it. Log every bug or surprising result in
   `docs/FINDINGS.md` with the numbers; I must be able to report them in the thesis.
4. Never say something works without running it. Show the real command output
   and numbers after each module (tests, smoke test from 12.1 S8).
5. Never touch the final test period during optimization. Never fake,
   hard-code or cherry-pick results.
6. MOMFA and all baselines use the same split, folds, training protocol,
   evaluation budget and seeds.
7. Save results in the format in Section 10. Generate plots only from
   saved files (Section 11), with `plots/make_figures.py`.
8. After each experiment, run the sanity checks in 12.1 and report
   pass/fail in plain language.
9. Follow the build order in Section 13 R7 and confirm each step with me
   before moving to the next.

## Stack

Python 3, PyTorch, pandas-ta, NumPy, pandas, SciPy, tvDatafeed.
