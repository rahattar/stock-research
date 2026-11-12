# Rebuilding the data (raw files are not in the repo)

Run from the project root. Needs Python with yfinance, and curl.

ETFs (data/etf/):
    mkdir -p data/etf
    python - <<'EOF'
    import yfinance as yf
    for t in ["SPY","IWM","EFA","EEM","IEF","TLT","LQD","HYG","GLD","DBC","VNQ","FXE","FXY","UUP"]:
        yf.Ticker(t).history(start="2000-01-01", end="2025-11-01", auto_adjust=False).to_csv(f"data/etf/{t.lower()}.csv")
    EOF

Funds, task 14 fresh fold (data/fund/):
    mkdir -p data/fund
    python - <<'EOF'
    import yfinance as yf
    for t in ["VFINX", "VFITX"]:
        yf.Ticker(t).history(start="1991-01-01", end="2005-10-01", auto_adjust=False).to_csv(f"data/fund/{t.lower()}.csv")
    EOF

Cash rate (data/fred/dtb3.csv):
    mkdir -p data/fred
    curl -sL -o data/fred/dtb3.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

If the files already exist in the forex-research folder, copy them instead:
    sh scripts/import_from_forex.sh /path/to/forex

Locked test period: 2025-11-01 to 2026-10-31. Do not download or use it except for one final run.
