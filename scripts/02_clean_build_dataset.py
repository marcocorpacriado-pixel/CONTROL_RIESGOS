from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def common_start_all_columns(df: pd.DataFrame) -> pd.Timestamp:
    first_valids = df.apply(lambda s: s.first_valid_index())
    common_start = first_valids.max()
    if pd.isna(common_start):
        raise RuntimeError("No se puede determinar common_start (todo NaN).")
    return pd.to_datetime(common_start)


def align_fred_to_index(fred: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    # FRED no siempre tiene datos en festivos -> forward fill sobre calendario de trading
    return fred.reindex(target_index).ffill()


def main() -> None:
    ensure_dirs()

    prices_path = RAW_DIR / "prices_yf.parquet"
    fred_path = RAW_DIR / "fred.parquet"

    if not prices_path.exists():
        raise FileNotFoundError(f"No existe {prices_path}. Ejecuta 01_download_data.py primero.")
    if not fred_path.exists():
        raise FileNotFoundError(f"No existe {fred_path}. Ejecuta 01_download_data.py primero.")

    prices = pd.read_parquet(prices_path).sort_index()
    fred = pd.read_parquet(fred_path).sort_index()

    prices.index = pd.to_datetime(prices.index)
    fred.index = pd.to_datetime(fred.index)

    # Separación: SP500 para régimen vs activos de cartera
    sp_col = "GSPC"
    if sp_col not in prices.columns:
        raise RuntimeError(
            "No encuentro la columna GSPC en prices_yf.parquet. "
            "Asegúrate de que en 01_download_data.py renombraste ^GSPC -> GSPC."
        )

    prices_sp = prices[[sp_col]].dropna()
    prices_assets = prices.drop(columns=[sp_col], errors="ignore")

    # ============================================================
    # 1) DATASET DE CARTERA (multivariante): intersección completa
    # ============================================================
    # Recorta al inicio común de todos los activos (evita NaNs por IPOs)
    assets_common_start = common_start_all_columns(prices_assets)
    prices_assets_common = prices_assets.loc[assets_common_start:].copy()

    # Intersección de fechas (todas las columnas presentes)
    prices_assets_common = prices_assets_common.dropna(axis=0, how="any")

    # Retornos log
    rets_assets_log = compute_log_returns(prices_assets_common).dropna()

    # ============================================================
    # 2) DATASET DE RÉGIMEN (largo): SP500 + FRED alineado
    # ============================================================
    # Retornos del SP500 (log)
    rets_sp_log = compute_log_returns(prices_sp).dropna()

    # Alinea FRED a los días de trading del SP500
    fred_aligned = align_fred_to_index(fred, rets_sp_log.index)

    # Features macro (transformaciones “limpieza”: bp, cambios diarios)
    feats = pd.DataFrame(index=rets_sp_log.index)
    feats["sp_ret_log"] = rets_sp_log[sp_col]

    if "VIXCLS" in fred_aligned.columns:
        feats["vix"] = fred_aligned["VIXCLS"]
        feats["dvix"] = feats["vix"].diff()

    if "DGS10" in fred_aligned.columns:
        feats["dgs10_bp"] = fred_aligned["DGS10"] * 100  # % -> bp
        feats["ddgs10_bp"] = feats["dgs10_bp"].diff()

    if "DGS2" in fred_aligned.columns:
        feats["dgs2_bp"] = fred_aligned["DGS2"] * 100
        feats["ddgs2_bp"] = feats["dgs2_bp"].diff()

    if "BAMLH0A0HYM2" in fred_aligned.columns:
        feats["hy_oas_bp"] = fred_aligned["BAMLH0A0HYM2"] * 100
        feats["dhy_oas_bp"] = feats["hy_oas_bp"].diff()

    # Limpieza final: quita primeras filas con diff NaN
    feats_regime = feats.dropna()

    # ============================================================
    # 3) Reportes de calidad (missing, rangos, tamaños)
    # ============================================================
    coverage_assets = pd.DataFrame({
        "first_valid": prices_assets.apply(lambda s: s.first_valid_index()),
        "last_valid": prices_assets.apply(lambda s: s.last_valid_index()),
        "n_missing_total": prices_assets.isna().sum(),
    })

    summary = {
        "assets_common_start": str(assets_common_start.date()),
        "assets_days_prices_common": int(prices_assets_common.shape[0]),
        "assets_days_returns": int(rets_assets_log.shape[0]),
        "n_assets": int(prices_assets_common.shape[1]),
        "regime_start": str(feats_regime.index.min().date()),
        "regime_end": str(feats_regime.index.max().date()),
        "regime_days": int(feats_regime.shape[0]),
        "regime_features": list(feats_regime.columns),
    }

    # ============================================================
    # 4) Guardado (parquet + csv)
    # ============================================================
    # Cartera
    prices_assets_common.to_parquet(PROCESSED_DIR / "prices_assets_common.parquet")
    rets_assets_log.to_parquet(PROCESSED_DIR / "returns_assets_log.parquet")

    prices_assets_common.to_csv(PROCESSED_DIR / "prices_assets_common.csv")
    rets_assets_log.to_csv(PROCESSED_DIR / "returns_assets_log.csv")

    # Régimen
    prices_sp.to_parquet(PROCESSED_DIR / "sp500_prices.parquet")
    rets_sp_log.to_parquet(PROCESSED_DIR / "sp500_returns_log.parquet")
    fred_aligned.to_parquet(PROCESSED_DIR / "fred_aligned_to_sp500.parquet")
    feats_regime.to_parquet(PROCESSED_DIR / "regime_features.parquet")

    prices_sp.to_csv(PROCESSED_DIR / "sp500_prices.csv")
    rets_sp_log.to_csv(PROCESSED_DIR / "sp500_returns_log.csv")
    feats_regime.to_csv(PROCESSED_DIR / "regime_features.csv")

    # Reportes
    coverage_assets.to_csv(PROCESSED_DIR / "coverage_assets_raw.csv")
    with open(PROCESSED_DIR / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("CLEAN BUILD OK")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()