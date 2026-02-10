from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    figures_dir: Path = project_root / "reports" / "figures"

    # Time horizon / simulation
    n_sims: int = 10_000
    horizon_days: int = 126  # ~6 meses de trading days

    # Reproducibility
    seed: int = 42

    # Regime model
    n_states: int = 2  # calma / crisis

    # Example universe (lo ajustarás luego)
    equity_tickers: tuple[str, ...] = (
        "AAPL","AMZN","BAC","BRK-B","CVX","ENPH","GME","GOOGL",
        "JNJ","JPM","MSFT","NVDA","PG","XOM"
    )
    etf_tickers: tuple[str, ...] = ("GLD","HYG")

settings = Settings()