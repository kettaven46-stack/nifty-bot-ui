import requests
import pandas as pd
from urllib.parse import quote
from datetime import datetime, timedelta

# ============ SETTINGS ============
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2UEE2QzUiLCJqdGkiOiI2OTgxZWY0MWM2Y2ExNjU4ZTU3NGRlMGIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcwMTIzMDczLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzAxNTYwMDB9.1-Yy59WeUn1nWKlmgf21Zy3A6C_yHFUWZowkuY_FLS8"     # keep private
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
INTERVAL = "1minute"

# Choose your date range here (YYYY-MM-DD)
FROM_DATE = "2024-01-01"
TO_DATE   = "2024-01-10"

# If Upstox restricts range, reduce CHUNK_DAYS (try 3, 2, or 1)
CHUNK_DAYS = 3

# Output file
OUT_XLSX = "nifty_1minute_raw.xlsx"

# ============ HELPERS ============
def fetch_chunk(from_date: str, to_date: str) -> pd.DataFrame:
    encoded_instrument = quote(INSTRUMENT_KEY, safe="")
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_instrument}/{INTERVAL}/{to_date}/{from_date}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

    data = r.json()
    candles = data["data"]["candles"]  # [time, open, high, low, close, volume, oi]

    df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume","oi"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def daterange_chunks(start_dt: datetime, end_dt: datetime, chunk_days: int):
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_dt)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)

# ============ MAIN ============
def main():
    start_dt = datetime.strptime(FROM_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(TO_DATE, "%Y-%m-%d")

    all_parts = []
    for f, t in daterange_chunks(start_dt, end_dt, CHUNK_DAYS):
        print(f"Fetching: {f} → {t}")
        part = fetch_chunk(f, t)
        all_parts.append(part)

    df = pd.concat(all_parts, ignore_index=True)

    # Clean & sort
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    df.to_excel(OUT_XLSX, index=False)
    print("SUCCESS ✅ Saved:", OUT_XLSX)
    print("Rows:", len(df))
    print(df.head(3))

if __name__ == "__main__":
    main()
