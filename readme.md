# stock-research

Pre-registered tests of a levered stock/bond portfolio with risk control, for a small account.
Split from forex-research on 2026-11-11.

Start with summary.md: goal, methodology, every result, open issues and the plan.

- prereg_task14.md: frozen specification of the current task.
- src/, scripts/, tests/: Python 3, pandas, numpy, pytest.
- results/: outputs of every task.
- data/readme.md: how to rebuild the data folders (raw data is not in the repo).
- scripts/import_from_forex.sh: copies the task 14 code, results and local data from the
  forex-research folder on the owner's machine.

Run tests: python -m pytest -q
