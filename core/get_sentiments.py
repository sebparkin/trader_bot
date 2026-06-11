import os
import time
import calendar
import requests
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["ALPHA_VANTAGE_KEY"]
BASE_URL = "https://www.alphavantage.co/query"
TICKERS = ["AAPL", "MSFT"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment")
DELAY_SECONDS = 2


def month_range(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(year, month):
    last_day = calendar.monthrange(year, month)[1]
    time_from = f"{year}{month:02d}01T0000"
    time_to = f"{year}{month:02d}{last_day:02d}T2359"

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(TICKERS),
        "time_from": time_from,
        "time_to": time_to,
        "limit": 1000,
        "sort": "EARLIEST",
        "apikey": API_KEY,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Note" in data or "Information" in data:
        msg = data.get("Note") or data.get("Information")
        raise RuntimeError(f"API limit reached: {msg}")

    return data.get("feed", [])


def aggregate_daily(feed, ticker):
    records = []
    for article in feed:
        pub_date = article.get("time_published", "")[:8]
        for ts in article.get("ticker_sentiment", []):
            if ts["ticker"] == ticker:
                records.append({
                    "Date": pub_date,
                    "score": float(ts["ticker_sentiment_score"]),
                    "relevance": float(ts["relevance_score"]),
                })

    if not records:
        return pd.DataFrame(columns=["Date", "SentimentMean", "SentimentWeighted", "ArticleCount"])

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d").dt.date

    def weighted_mean(g):
        total_rel = g["relevance"].sum()
        if total_rel == 0:
            return g["score"].mean()
        return (g["score"] * g["relevance"]).sum() / total_rel

    daily = df.groupby("Date").agg(
        SentimentMean=("score", "mean"),
        ArticleCount=("score", "count"),
    )
    daily["SentimentWeighted"] = df.groupby("Date").apply(weighted_mean, include_groups=False)
    return daily.reset_index()


def update_sentiment(ticker):
    """Fetch any missing recent sentiment data and append it to the CSV."""
    path = os.path.join(OUTPUT_DIR, f"sentiment_{ticker}.csv")

    if not os.path.exists(path):
        print(f"No sentiment CSV for {ticker} — run get_sentiments.py first to fetch historical data")
        return

    existing = pd.read_csv(path)
    existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
    last_date = existing["Date"].max()

    today = date.today()
    weekday = today.weekday()
    if weekday == 0:
        yesterday = today - timedelta(days=3)
    elif weekday == 6:
        yesterday = today - timedelta(days=2)
    else:
        yesterday = today - timedelta(days=1)

    if last_date >= yesterday:
        return  # Already up to date

    print(f"Updating {ticker} sentiment: {last_date} → {yesterday}")

    next_day = last_date + timedelta(days=1)
    time_from = f"{next_day.year}{next_day.month:02d}{next_day.day:02d}T0000"
    time_to = f"{yesterday.year}{yesterday.month:02d}{yesterday.day:02d}T2359"

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "limit": 1000,
        "sort": "EARLIEST",
        "apikey": API_KEY,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            print(f"API limit hit — sentiment not updated: {msg}")
            return

        feed = data.get("feed", [])
        new_daily = aggregate_daily(feed, ticker)

        if new_daily.empty:
            print(f"No new articles found for {ticker}")
            return

        combined = pd.concat([existing, new_daily]).sort_values("Date").drop_duplicates("Date")
        combined.to_csv(path, index=False)
        print(f"Added {len(new_daily)} new day(s) for {ticker}")

    except Exception as e:
        print(f"Failed to update sentiment for {ticker}: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = date.today()
    two_years_ago = date(today.year - 2, today.month, 1)

    months = list(month_range(two_years_ago.year, two_years_ago.month, today.year, today.month))
    print(f"Fetching {len(months)} months for {TICKERS}  ({len(months)} API requests)")
    print(f"Free tier limit is 25 requests/day — this will complete in one run.\n")

    all_data = {ticker: [] for ticker in TICKERS}

    for i, (year, month) in enumerate(months):
        print(f"[{i+1}/{len(months)}] {year}-{month:02d} ...", end=" ", flush=True)

        try:
            feed = fetch_month(year, month)
            print(f"{len(feed)} articles")

            for ticker in TICKERS:
                daily = aggregate_daily(feed, ticker)
                if not daily.empty:
                    all_data[ticker].append(daily)

        except RuntimeError as e:
            print(f"\nStopped early: {e}")
            print("Partial results will be saved. Run again tomorrow to continue.")
            break
        except Exception as e:
            print(f"Error ({e}) — skipping")

        if i < len(months) - 1:
            time.sleep(DELAY_SECONDS)

    print()
    for ticker in TICKERS:
        chunks = all_data[ticker]
        if not chunks:
            print(f"No data collected for {ticker}")
            continue

        df = pd.concat(chunks).sort_values("Date").drop_duplicates("Date")
        path = os.path.join(OUTPUT_DIR, f"sentiment_{ticker}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} daily rows for {ticker}  →  {path}")


if __name__ == "__main__":
    main()
