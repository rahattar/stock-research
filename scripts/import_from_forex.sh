#!/bin/sh
# Copy the task 14 code, tests, results and the local data from the forex-research folder.
# Run from the stock-research root:  sh scripts/import_from_forex.sh /path/to/forex
set -e
from="$1"
if [ -z "$from" ] || [ ! -f "$from/src/vol_target.py" ]; then
    echo "usage: sh scripts/import_from_forex.sh /path/to/forex   (folder that holds src/vol_target.py)"
    exit 1
fi
for f in src/vol_target.py src/vol_target_data.py scripts/run_vol_target.py tests/test_vol_target.py; do
    cp "$from/$f" "$f"
    echo "copied $f"
done
cp "$from"/results/vol_target_* results/
echo "copied results/vol_target_*"
mkdir -p data/etf data/fund data/fred
cp "$from"/data/etf/*.csv data/etf/
cp "$from"/data/fund/*.csv data/fund/
cp "$from"/data/fred/dtb3.csv data/fred/
echo "copied data/etf, data/fund, data/fred/dtb3.csv (not tracked by git)"
echo "next: python -m pytest -q, then git add -A, git commit, git push"
