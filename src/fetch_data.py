"""Pull company data from vnstock and save it as raw JSON documents.

Run with: python src/fetch_data.py
"""
import json
from pathlib import Path

from vnstock import Company

TICKERS = ["FPT", "VNM", "VIC", "HPG", "MWG", "VCB", "MSN", "GAS", "VHM", "TCB"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_overview(symbol: str) -> dict:
    df = Company(symbol=symbol).overview()
    return df.iloc[0].to_dict()


def fetch_news(symbol: str, limit: int = 10) -> list[dict]:
    df = Company(symbol=symbol).news()
    return df.head(limit).to_dict(orient="records")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in TICKERS:
        print(f"Fetching {symbol}...")

        overview = fetch_overview(symbol)
        (RAW_DIR / f"{symbol}_overview.json").write_text(
            json.dumps(overview, ensure_ascii=False, indent=2, default=str)
        )

        news = fetch_news(symbol)
        (RAW_DIR / f"{symbol}_news.json").write_text(
            json.dumps(news, ensure_ascii=False, indent=2, default=str)
        )

    print(f"Done. Saved data for {len(TICKERS)} companies to {RAW_DIR}")


if __name__ == "__main__":
    main()
