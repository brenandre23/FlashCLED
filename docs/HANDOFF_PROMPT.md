# Handoff prompt

Copy everything inside the block below into a new Claude Code session, started on Opus 5.5 in a fresh clone of this repository with the branch `claude/new-session-7y01je` checked out. Fill in the two bracketed lines first.

```text
You are the manager for the FlashCLED work described in docs/IMPLEMENTATION_PLAN.md. Read that file in full before doing anything else, then read tests/test_no_leakage.py and pipeline/modeling/temporal_splits.py so you know the leakage rules the code now enforces.

Background. FlashCLED is a conflict early-warning pipeline for the Central African Republic: about 3,407 H3 resolution-5 cells on a 14-day spine from 2000, a two-stage hurdle ensemble (onset, then fatalities), conformal intervals, and a Flask dashboard. It was my master's thesis. An earlier session audited the code, found and fixed several sources of temporal leakage, changed parts of the methodology, and wrote the plan. Those fixes were tested only on synthetic data; nothing has been run against a real database yet. Reported skill will probably fall relative to the thesis figures, and that is expected.

State of this machine. My previous computer was wiped. The database, data/, models/, .env and configs/pruned_features.yaml were never in git.
USB backup: [none / describe what is on it and where it is mounted]
Credentials and manual input files: [ready in .env and data/raw / not yet]

What I want from you.
1. Check the machine against section 10 of the plan (Python, PostgreSQL with postgis, h3 and h3_postgis, .env, files in data/raw) and tell me exactly what is missing. Do not start a real pipeline run until I confirm.
2. Run Wave 1 of the plan now: it needs no data. Launch one Sonnet 5.5 worker per task, in parallel, each in its own git worktree (Agent tool, model "sonnet", isolation "worktree"). Give each worker its task, the exact files it owns, its acceptance check, and the invariants in section 3. Workers make no methodology decisions and do not touch shared files.
3. Review every worker's diff yourself. Merge one at a time. After each merge run `python -m pytest tests -q` and `ruff check --select F,E9 .`. Do not merge a worker whose acceptance check you have not seen pass.
4. Then run Wave 2 the same way. For Wave 3, write the design and show it to me before any code.
5. Commit in small logical commits and push to this branch once tests pass. Do not open a pull request unless I ask.

Rules.
- You own every methodology decision. Decide on statistical grounds, record the decision and the reason in docs/METHODOLOGY_CHANGES.md, and tell me. Ask me only when the choice depends on something you cannot find out, such as what I want to report.
- Before asking me a factual question about the code or the data, try to answer it yourself.
- Never drop tables, delete models or overwrite predictions without telling me first. Stale models are moved to models/superseded/, not deleted.
- Report honestly: say what was tested and how, and what was not.

Start by telling me, in a few lines, what you found on this machine and which workers you are launching.
```

## Notes

- The thesis PDF is not in the repository. Give the new session its path if you want it checked against the code.
- To reproduce the thesis split with the leakage fixes applied, set `split.train_cutoff` to `2024-01-01` in `configs/data.yaml`.
