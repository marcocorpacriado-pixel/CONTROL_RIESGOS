from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import os
from dotenv import load_dotenv
from fredapi import Fred


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_yfinance_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios ajustados (auto_adjust=True) y devuelve un DF wide: index=Date, cols=tickers.
    """
    # yfinance acepta lista o string con espacios
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if df.empty:
        raise RuntimeError("yfinance devolvió un DataFrame vacío. Revisa tickers o conexión.")

    # Caso 1: un solo ticker -> columnas normales (Open/High/Low/Close/Volume)
    # Caso 2: múltiples tickers -> columnas MultiIndex (Ticker, Field)
    if isinstance(df.columns, pd.MultiIndex):
        # Nos quedamos con el "Close" ajustado
        closes = []
        for t in tickers:
            if (t, "Close") in df.columns:
                closes.append(df[(t, "Close")].rename(t))
            else:
                # algunos tickers pueden fallar
                closes.append(pd.Series(name=t, dtype="float64"))
        out = pd.concat(closes, axis=1)
    else:
        # Un ticker
        out = df["Close"].to_frame(name=tickers[0])

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def download_fred_series(series: list[str], start: str, end: str) -> pd.DataFrame:
    load_dotenv()  # carga .env si existe
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta FRED_API_KEY. Ponla en .env (NO se sube al repo) o como variable de entorno."
        )

    fred = Fred(api_key=api_key)
    frames = []
    for sid in series:
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        s.name = sid
        frames.append(s)

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def main(start: str, end: str) -> None:
    ensure_dirs()

    load_dotenv()  # carga .env si existe

    # Universo del enunciado
    equity_tickers = [
        "AAPL", "AMZN", "BAC", "BRK-B", "CVX", "ENPH", "GME", "GOOGL",
        "JNJ", "JPM", "MSFT", "NVDA", "PG", "XOM",
    ]
    etf_tickers = ["GLD", "HYG"]

    # Indicadores para régimen (S&P 500)
    regime_tickers = ["^GSPC"]  # S&P 500 index en yfinance

    # Descarga precios (activos + índice)
    price_tickers = equity_tickers + etf_tickers + regime_tickers

    print(f"[1/2] Descargando precios yfinance ({len(price_tickers)} tickers) ...")
    prices = download_yfinance_prices(price_tickers, start=start, end=end)

    # Renombrar ^GSPC a algo cómodo (evita caracteres raros)
    prices = prices.rename(columns={"^GSPC": "GSPC"})

    prices_path = RAW_DIR / "prices_yf.parquet"
    prices.to_parquet(prices_path)
    print(f"Guardado: {prices_path}")

    # Series FRED recomendadas (útiles para limpieza y luego para régimen)
    fred_series = [
        "DGS10",          # Treasury 10Y yield
        "DGS2",           # Treasury 2Y yield
        "VIXCLS",         # VIX
        "BAMLH0A0HYM2",   # ICE BofA US High Yield OAS (spread)
    ]

    print(f"[2/2] Descargando series FRED ({len(fred_series)} series) ...")
    fred = download_fred_series(fred_series, start=start, end=end)

    fred_path = RAW_DIR / "fred.parquet"
    fred.to_parquet(fred_path)
    print(f"Guardado: {fred_path}")

    # Log simple de cobertura
    coverage = pd.DataFrame({
        "first_valid": prices.apply(lambda s: s.first_valid_index()),
        "last_valid": prices.apply(lambda s: s.last_valid_index()),
        "n_missing": prices.isna().sum(),
    })
    cov_path = RAW_DIR / "coverage_prices.csv"
    coverage.to_csv(cov_path, index=True)
    print(f"Guardado: {cov_path}")

    print("OK — descarga finalizada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2006-01-01")
    parser.add_argument("--end", type=str, default=str(date.today()))
    args = parser.parse_args()

    main(start=args.start, end=args.end)