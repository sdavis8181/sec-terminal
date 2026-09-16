import os
import re
import warnings
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="SEC XBRL Financial Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Institutional SEC XBRL Financial Terminal")
st.markdown(
    "Quarterly financial data directly from SEC XBRL (US-GAAP & IFRS), "
    "with Yahoo Finance for historical pricing and secondary fallback."
)

DEFAULT_IDENTITY = "FinancialTerminal User@example.com"
try:
    SEC_IDENTITY = st.secrets.get("EDGAR_IDENTITY", DEFAULT_IDENTITY)
except Exception:
    SEC_IDENTITY = os.getenv("EDGAR_IDENTITY", DEFAULT_IDENTITY)

if not SEC_IDENTITY or "@" not in SEC_IDENTITY:
    SEC_IDENTITY = DEFAULT_IDENTITY

SEC_HEADERS = {
    "User-Agent": SEC_IDENTITY,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKER_HEADERS = {
    "User-Agent": SEC_IDENTITY,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.header("Terminal Controls")

    with st.form(key="terminal_controls"):
        ticker_symbol = (
            st.text_input("Stock Ticker", value="TMDX", max_chars=12)
            .strip()
            .upper()
        )

        lookback_quarters = st.slider(
            "Historical Lookback (Quarters)",
            min_value=8,
            max_value=40,
            value=20,
            step=4,
        )

        force_refresh = st.checkbox("Bypass Cache / Force Refresh", value=False)
        run_button = st.form_submit_button("Generate Report", type="primary")

    st.markdown("---")
    st.caption(
        "Direct 3-month standalone quarterly reports take precedence, with "
        "cumulative flow subtractions ($6\\text{M}-3\\text{M}$, $9\\text{M}-6\\text{M}$, "
        "$12\\text{M}-9\\text{M}$) backing out standalone Q4 numbers across changing taxonomy tags. "
        "All periods are then snapped onto a strict calendar-quarter grid so YoY/TTM math never "
        "silently bridges a missing quarter."
    )

# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_number(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if pd.notna(value) else np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat", "n/a", "na", "-"}:
        return np.nan

    negative = s.startswith("(") and s.endswith(")")
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "")

    try:
        x = float(s)
        return -x if negative else x
    except Exception:
        return np.nan


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


def pct_change_safe(series, periods):
    return series.pct_change(periods=periods, fill_method=None) * 100


def as_date(value):
    return pd.to_datetime(value, errors="coerce")


def canonical_quarter_end(date):
    """
    Snap any date onto the calendar-quarter-end grid (3/31, 6/30, 9/30, 12/31).

    Different XBRL concepts (Revenue vs OCF vs EPS, etc.) frequently report the
    "same" quarter with end dates that differ by a day or two (fiscal calendar
    rounding, 52/53-week quirks, restatement filings). Left as-is, those near
    duplicates land in different rows of the fact table, so one field ends up
    populated and another ends up NaN for what is really the same quarter -
    this is the main reason the Revenue series had holes that other fields
    didn't. Canonicalizing every period onto the same quarter-end key forces
    all fields for one true fiscal quarter to merge into a single row.
    """
    d = pd.Timestamp(date)
    if pd.isna(d):
        return pd.NaT
    return d.to_period("Q").end_time.normalize()


# ============================================================
# SEC EDGAR APIS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_sec_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=TICKER_HEADERS, timeout=30)
    response.raise_for_status()

    raw = response.json()
    rows = [
        {
            "ticker": str(item.get("ticker", "")).upper(),
            "title": item.get("title", ""),
            "cik": int(item.get("cik_str", 0)),
        }
        for item in raw.values()
    ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def get_cik_for_ticker(ticker):
    df = get_sec_ticker_map()
    match = df[df["ticker"].eq(ticker.upper())]
    if match.empty:
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR registries.")
    return int(match.iloc[0]["cik"])


@st.cache_data(ttl=86400, show_spinner=False)
def get_companyfacts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    response = requests.get(url, headers=SEC_HEADERS, timeout=45)
    response.raise_for_status()
    return response.json()


# ============================================================
# XBRL CONCEPT DEFINITIONS
# ============================================================

FLOW_CONCEPTS = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractsWithCustomers",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "Revenue",
    ],
    "Operating_Income": [
        "OperatingIncomeLoss",
        "OperatingProfitLoss",
        "ProfitLossFromOperatingActivities",
    ],
    "Net_Income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "ProfitLossAttributableToOwners",
    ],
    "Diluted_EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "DilutedEarningsPerShare",
    ],
    "Diluted_Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ],
    "OCF": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashFlowsFromUsedInOperatingActivities",
    ],
    "Capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PurchaseOfPropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "AdditionsToPropertyPlantAndEquipment",
        "PaymentsForSoftware",
        "PaymentsToAcquireIntangibleAssets",
    ],
}

FLOW_FIELDS = [
    "Revenue",
    "Operating_Income",
    "Net_Income",
    "Diluted_EPS",
    "Diluted_Shares",
    "OCF",
    "Capex",
]


# ============================================================
# XBRL FACT PARSER
# ============================================================

def all_fact_candidates(companyfacts, field):
    facts = companyfacts.get("facts", {})
    preferred = FLOW_CONCEPTS.get(field, [])
    candidates = []

    for taxonomy, taxonomy_facts in facts.items():
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept in preferred:
            if concept in taxonomy_facts:
                candidates.append((taxonomy, concept, taxonomy_facts[concept]))
    return candidates


def choose_unit_name(concept_data, field):
    units = concept_data.get("units", {})
    if not units:
        return None

    if field == "Diluted_EPS":
        for unit in ["USD/shares", "USD / shares", "EUR/shares", "ARS/shares"]:
            if unit in units:
                return unit

    if field == "Diluted_Shares":
        if "shares" in units:
            return "shares"

    for unit in ["USD", "EUR", "GBP", "CAD", "BRL", "ARS"]:
        if unit in units:
            return unit

    return next(iter(units.keys())) if len(units) == 1 else None


def get_all_usable_entries(companyfacts, field):
    """
    Collects entries across all valid taxonomy concepts to handle concept shifts
    between fiscal years and reporting periods.
    """
    candidates = all_fact_candidates(companyfacts, field)
    all_entries = []

    for taxonomy, concept, concept_data in candidates:
        unit = choose_unit_name(concept_data, field)
        if not unit:
            continue
        entries = concept_data["units"].get(unit, [])
        usable = [
            dict(x, taxonomy=taxonomy, concept=concept, unit=unit)
            for x in entries
            if isinstance(x, dict) and "val" in x
        ]
        all_entries.extend(usable)

    return all_entries


def entry_date(entry, key):
    return as_date(entry.get(key))


def entry_duration_days(entry):
    start = entry_date(entry, "start")
    end = entry_date(entry, "end")
    if pd.isna(start) or pd.isna(end):
        return None
    return (end - start).days + 1


def quarter_from_frame(frame):
    if not frame:
        return None
    match = re.search(r"CY(\d{4})Q([1-4])", str(frame))
    if not match:
        return None
    return pd.Timestamp(
        year=int(match.group(1)),
        month=int(match.group(2)) * 3,
        day=1,
    ) + pd.offsets.MonthEnd(0)


def standalone_quarter_candidates(entries):
    output = []
    for e in entries:
        end = entry_date(e, "end")
        if pd.isna(end):
            continue
        form = str(e.get("form", "")).upper()
        frame = str(e.get("frame", ""))

        if "start" in e and "end" in e:
            days = entry_duration_days(e)
            if days is not None and 70 <= days <= 110:
                score = 60 + (20 if form in {"10-Q", "20-F", "6-K"} else 0)
                output.append({"entry": e, "period": end, "method": "duration", "score": score})
                continue

            q_end = quarter_from_frame(frame)
            if q_end is not None:
                output.append({"entry": e, "period": q_end, "method": "SEC frame", "score": 50})
        else:
            score = 40 + (20 if form in {"10-Q", "10-K"} else 0)
            output.append({"entry": e, "period": end, "method": "instant", "score": score})

    return output


def pick_best_entry(candidates):
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda x: (
            str(x["entry"].get("filed", "") if "entry" in x else x.get("filed", "")),
            str(x["entry"].get("accn", "") if "entry" in x else x.get("accn", "")),
        ),
    )[-1]


def derive_quarterly_flows(entries, field):
    """
    Backs out standalone quarters (especially Q4) from cumulative YTD facts:
    Q1 = direct 3M figure, Q2 = 6M - Q1, Q3 = 9M - 6M, Q4 = 12M - 9M.

    IMPORTANT FIX: this used to group entries by the SEC-provided "fy" tag.
    That tag describes the *filing's own* fiscal-year context, not the actual
    period the fact covers - the exact same fact often reappears in a later
    filing (as a prior-year comparative) tagged with a *different* fy value.
    Grouping on "fy" therefore scattered a single fiscal year's Q1/6M/9M/12M
    figures across multiple buckets, so the subtraction logic silently failed
    and produced holes (this was the main cause of the missing Revenue
    quarters). Every duration within one fiscal year (Q1, 6M, 9M, 12M) shares
    the exact same period *start* date, so we group on that instead - it's a
    property of the period itself, not of whichever filing happened to report it.
    """
    usable = []
    for e in entries:
        if "start" not in e or "end" not in e:
            continue
        days = entry_duration_days(e)
        val = clean_number(e.get("val"))
        start = entry_date(e, "start")
        end = entry_date(e, "end")
        filed = str(e.get("filed", ""))

        if days is not None and pd.notna(val) and pd.notna(start) and pd.notna(end):
            usable.append({
                "entry": e, "val": val, "days": days,
                "start": start, "end": end, "filed": filed
            })

    if not usable:
        return {}

    df_entries = pd.DataFrame(usable).sort_values(["start", "end", "filed"])
    derived = {}

    for _, group in df_entries.groupby("start"):
        q1_rows = group[(group["days"] >= 70) & (group["days"] <= 110)]
        m6_rows = group[(group["days"] >= 160) & (group["days"] <= 205)]
        m9_rows = group[(group["days"] >= 250) & (group["days"] <= 300)]
        m12_rows = group[(group["days"] >= 340) & (group["days"] <= 385)]

        if not q1_rows.empty:
            q1 = q1_rows.iloc[-1]
            key = canonical_quarter_end(q1["end"])
            derived[key.strftime("%Y-%m-%d")] = {
                "val": q1["val"], "period": key, "method": "direct Q1", "entry": q1["entry"]
            }

        if not m6_rows.empty and not q1_rows.empty:
            m6 = m6_rows.iloc[-1]
            q1 = q1_rows.iloc[-1]
            key = canonical_quarter_end(m6["end"])
            derived[key.strftime("%Y-%m-%d")] = {
                "val": m6["val"] - q1["val"], "period": key, "method": "derived (6M - Q1)", "entry": m6["entry"]
            }

        if not m9_rows.empty and not m6_rows.empty:
            m9 = m9_rows.iloc[-1]
            m6 = m6_rows.iloc[-1]
            key = canonical_quarter_end(m9["end"])
            derived[key.strftime("%Y-%m-%d")] = {
                "val": m9["val"] - m6["val"], "period": key, "method": "derived (9M - 6M)", "entry": m9["entry"]
            }

        if not m12_rows.empty:
            m12 = m12_rows.iloc[-1]
            val_q4 = np.nan
            if not m9_rows.empty:
                val_q4 = m12["val"] - m9_rows.iloc[-1]["val"]
            elif len(q1_rows) >= 1 and not m6_rows.empty:
                val_q4 = m12["val"] - m6_rows.iloc[-1]["val"]

            if pd.notna(val_q4):
                key = canonical_quarter_end(m12["end"])
                derived[key.strftime("%Y-%m-%d")] = {
                    "val": val_q4, "period": key, "method": "derived Q4 (12M - 9M)", "entry": m12["entry"]
                }

    return derived


# ============================================================
# DATA MERGING & METRICS
# ============================================================

def collapse_duplicate_quarters(df, tolerance_days=45):
    if df.empty or "Period" not in df.columns:
        return df

    df = df.copy()
    df["Period"] = pd.to_datetime(df["Period"], errors="coerce")
    df = df.dropna(subset=["Period"]).sort_values("Period").reset_index(drop=True)

    clusters = []
    for _, row in df.iterrows():
        p = row["Period"]
        placed = False
        for cluster in clusters:
            if any(abs((p - existing["Period"]).days) <= tolerance_days for existing in cluster):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])

    unified_rows = []
    for cluster in clusters:
        sec_rows = [r for r in cluster if "SEC" in str(r.get("Source", ""))]
        canonical_row = sec_rows[-1] if sec_rows else cluster[-1]
        canonical_period = canonical_row["Period"]

        merged_record = {"Period": canonical_period}
        for col in df.columns:
            if col == "Period":
                continue
            val = np.nan
            for r in sec_rows:
                if pd.notna(r.get(col)):
                    val = r[col]
            if pd.isna(val):
                for r in cluster:
                    if pd.notna(r.get(col)):
                        val = r[col]
            merged_record[col] = val

        sources = {str(r.get("Source", "")) for r in cluster if pd.notna(r.get("Source"))}
        if any("SEC" in s for s in sources) and any("Yahoo" in s for s in sources):
            merged_record["Source"] = "SEC + Yahoo"
        elif any("SEC" in s for s in sources):
            merged_record["Source"] = "SEC XBRL"
        else:
            merged_record["Source"] = "Yahoo Finance"

        unified_rows.append(merged_record)

    return pd.DataFrame(unified_rows).sort_values("Period").reset_index(drop=True)


def reindex_to_quarterly(df):
    """
    Snap the merged fact table onto a *strict, gap-explicit* calendar-quarter
    grid (one row per quarter-end, from the earliest to the latest period).

    Without this, a fiscal quarter with no data at all simply doesn't exist as
    a row, so pandas' pct_change(periods=4) and rolling(4) window operations
    silently compare/sum whatever rows *are* present - even if they aren't
    actually 4 consecutive quarters apart. That's what made the P/S (and P/E,
    FCF yield) charts "skip" periods and show misleading TTM figures instead
    of a clean gap: the rolling window was quietly bridging over a hole. Once
    every real quarter has an explicit (possibly all-NaN) row, TTM/YoY math
    only ever operates over genuinely consecutive quarters.
    """
    if df.empty or "Period" not in df.columns:
        return df

    df = df.copy()
    df["Period"] = pd.to_datetime(df["Period"], errors="coerce")
    df = df.dropna(subset=["Period"]).sort_values("Period")

    full_index = pd.date_range(start=df["Period"].min(), end=df["Period"].max(), freq="Q")
    df = df.set_index("Period").reindex(full_index).rename_axis("Period").reset_index()

    if "Source" in df.columns:
        df["Source"] = df["Source"].fillna("No filing found")

    return df


def build_quarterly_fact_table(companyfacts):
    fact_meta = {}
    quarter_rows = {}

    for field in FLOW_FIELDS:
        entries = get_all_usable_entries(companyfacts, field)
        if not entries:
            fact_meta[field] = None
            continue

        fact_meta[field] = {
            "taxonomy": entries[0]["taxonomy"],
            "concept": entries[0]["concept"],
            "unit": entries[0]["unit"],
        }

        # Step 1: Populate direct standalone 3-month reported quarters
        candidates = standalone_quarter_candidates(entries)
        grouped = {}
        for item in candidates:
            p = item["period"]
            if pd.isna(p):
                continue
            grouped.setdefault(p.strftime("%Y-%m-%d"), []).append(item)

        for key, group in grouped.items():
            best = pick_best_entry(group)
            if best is None:
                continue
            entry = best["entry"]
            canon_period = canonical_quarter_end(best["period"])
            canon_key = canon_period.strftime("%Y-%m-%d")
            quarter_rows.setdefault(canon_key, {"Period": canon_period, "Source": "SEC XBRL"})
            quarter_rows[canon_key][field] = clean_number(entry.get("val"))
            quarter_rows[canon_key][f"{field}_Concept"] = f"{entry.get('taxonomy')}:{entry.get('concept')}"
            quarter_rows[canon_key][f"{field}_Method"] = best["method"]

        # Step 2: Backfill missing quarters (e.g., Q4 or cumulative filers) via flow derivation
        if field in {"OCF", "Capex", "Revenue", "Operating_Income", "Net_Income"}:
            derived_map = derive_quarterly_flows(entries, field)
            for key, item in derived_map.items():
                quarter_rows.setdefault(key, {"Period": item["period"], "Source": "SEC XBRL"})
                if field not in quarter_rows[key] or pd.isna(quarter_rows[key].get(field)):
                    quarter_rows[key][field] = item["val"]
                    quarter_rows[key][f"{field}_Concept"] = f"{item['entry'].get('taxonomy')}:{item['entry'].get('concept')}"
                    quarter_rows[key][f"{field}_Method"] = item["method"]

    df = pd.DataFrame(list(quarter_rows.values()))
    if df.empty:
        return df, fact_meta

    df = collapse_duplicate_quarters(df, tolerance_days=45)
    df = df[df["Period"] >= pd.Timestamp("2017-01-01")].reset_index(drop=True)
    return df, fact_meta


YF_KEYS = {
    "Revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "Operating_Income": ["Operating Income", "Operating Income Loss", "EBIT"],
    "Net_Income": ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"],
    "Diluted_EPS": ["Diluted EPS", "Diluted EPS From Continuing Operations"],
    "Diluted_Shares": ["Diluted Average Shares", "Diluted Average Shares Outstanding", "Basic Average Shares"],
    "OCF": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"],
    "Capex": ["Capital Expenditure", "Capital Expenditure Reported", "Purchase Of Property Plant And Equipment"],
}


def fetch_yahoo_quarterly(ticker):
    try:
        tk = yf.Ticker(ticker)
        inc = tk.quarterly_income_stmt
        cf = tk.quarterly_cashflow

        if inc is None or inc.empty:
            return pd.DataFrame(), ["Yahoo quarterly income statement empty"]

        rows = []
        for date_col in inc.columns:
            period = pd.to_datetime(date_col, errors="coerce")
            if pd.isna(period):
                continue

            def get_yf_val(df_source, keys):
                if df_source is None or df_source.empty:
                    return np.nan
                for k in keys:
                    if k in df_source.index:
                        return clean_number(df_source.loc[k, date_col])
                return np.nan

            rows.append({
                "Period": period,
                "Source": "Yahoo Finance",
                "Revenue": get_yf_val(inc, YF_KEYS["Revenue"]),
                "Operating_Income": get_yf_val(inc, YF_KEYS["Operating_Income"]),
                "Net_Income": get_yf_val(inc, YF_KEYS["Net_Income"]),
                "Diluted_EPS": get_yf_val(inc, YF_KEYS["Diluted_EPS"]),
                "Diluted_Shares": get_yf_val(inc, YF_KEYS["Diluted_Shares"]),
                "OCF": get_yf_val(cf, YF_KEYS["OCF"]),
                "Capex": get_yf_val(cf, YF_KEYS["Capex"]),
            })
        return pd.DataFrame(rows), []
    except Exception as exc:
        return pd.DataFrame(), [f"Yahoo error: {exc}"]


def calculate_metrics(df):
    work = df.copy()
    for col in FLOW_FIELDS:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Reconcile Shares & EPS
    missing_eps = work["Diluted_EPS"].isna() & work["Net_Income"].notna() & work["Diluted_Shares"].notna()
    work.loc[missing_eps, "Diluted_EPS"] = safe_divide(work["Net_Income"], work["Diluted_Shares"])

    implied_shares = safe_divide(work["Net_Income"], work["Diluted_EPS"])
    missing_shares = (
        work["Diluted_Shares"].isna()
        & implied_shares.notna()
        & (work["Diluted_EPS"].abs() > 0.005)
        & (implied_shares > 0)
    )
    work.loc[missing_shares, "Diluted_Shares"] = implied_shares[missing_shares]

    work["Diluted_Shares_M"] = work["Diluted_Shares"].replace(0, np.nan).ffill().bfill() / 1e6
    work["Revenue_B"] = work["Revenue"] / 1e9
    work["Rev_YoY_%"] = pct_change_safe(work["Revenue_B"], 4)
    work["Rev_QoQ_%"] = pct_change_safe(work["Revenue_B"], 1)

    work["EPS_YoY_%"] = pct_change_safe(work["Diluted_EPS"], 4).clip(-500, 500)
    work["EPS_QoQ_%"] = pct_change_safe(work["Diluted_EPS"], 1).clip(-500, 500)

    work["Op_Margin_%"] = safe_divide(work["Operating_Income"], work["Revenue"]) * 100
    work["Net_Margin_%"] = safe_divide(work["Net_Income"], work["Revenue"]) * 100
    work["Share_Dilution_YoY_%"] = pct_change_safe(work["Diluted_Shares_M"], 4).clip(-50, 50)

    # Free Cash Flow
    work["Capex_Positive"] = work["Capex"].abs().fillna(0.0)
    work["FCF"] = np.where(work["OCF"].notna(), work["OCF"] - work["Capex_Positive"], np.nan)
    work["FCF_B"] = work["FCF"] / 1e9
    work["FCF_YoY_%"] = pct_change_safe(work["FCF"], 4).clip(-500, 500)
    work["FCF_QoQ_%"] = pct_change_safe(work["FCF"], 1).clip(-500, 500)

    return work


def calculate_trailing_flow_strict(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=4, min_periods=4).sum()


# ============================================================
# MARKET PRICING
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        shares_outstanding = info.get("sharesOutstanding")
        market_cap = info.get("marketCap")

        hist = tk.history(period="10y", auto_adjust=False)

        if hist is not None and not hist.empty:
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.sort_index()
            if current_price is None:
                current_price = clean_number(hist["Close"].iloc[-1])
            hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()

        if market_cap is None and current_price is not None and shares_outstanding is not None:
            market_cap = current_price * shares_outstanding

        return (
            hist,
            clean_number(current_price),
            clean_number(market_cap),
            clean_number(shares_outstanding),
        )
    except Exception:
        return pd.DataFrame(), np.nan, np.nan, np.nan


# ============================================================
# SEC PIPELINE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_and_parse_ticker(ticker, refresh_nonce=0):
    diagnostics = []
    cik = get_cik_for_ticker(ticker)
    companyfacts = get_companyfacts(cik)

    sec_df, fact_meta = build_quarterly_fact_table(companyfacts)
    if sec_df.empty:
        diagnostics.append("SEC Company Facts did not yield standalone quarterly flow facts.")

    yahoo_df, yahoo_diag = fetch_yahoo_quarterly(ticker)
    diagnostics.extend(yahoo_diag)

    if sec_df.empty and yahoo_df.empty:
        raise ValueError(f"No quarterly data found for {ticker}.")

    combined = pd.concat([sec_df, yahoo_df], ignore_index=True)
    df = collapse_duplicate_quarters(combined, tolerance_days=45)
    # Snap onto a gap-explicit quarterly calendar BEFORE any YoY/TTM math is
    # computed, so pct_change/rolling never silently bridge a missing quarter.
    df = reindex_to_quarterly(df)
    df = calculate_metrics(df)

    return df, fact_meta, diagnostics


# ============================================================
# MAIN EXECUTION
# ============================================================

if ticker_symbol:
    try:
        nonce = np.random.randint(1, 1000000) if force_refresh else 0

        with st.spinner(f"Loading SEC XBRL statements for {ticker_symbol}..."):
            df_all, fact_meta, diagnostics = fetch_and_parse_ticker(ticker_symbol, refresh_nonce=nonce)
            hist_price, current_price, market_cap, shares_outstanding = fetch_market_data(ticker_symbol)

        df_raw = df_all.tail(lookback_quarters).reset_index(drop=True)

        # ----------------------------------------------------
        # TTM AND AS-OF HISTORICAL VALUATION
        # ----------------------------------------------------
        df_raw["TTM_Revenue"] = calculate_trailing_flow_strict(df_raw["Revenue"])
        df_raw["TTM_EPS"] = calculate_trailing_flow_strict(df_raw["Diluted_EPS"])
        df_raw["TTM_FCF"] = calculate_trailing_flow_strict(df_raw["FCF"])

        if hist_price is not None and not hist_price.empty:
            prices_at_quarter = []
            for q_date in df_raw["Period"]:
                historical_closes = hist_price.loc[hist_price.index <= q_date, "Close"]
                prices_at_quarter.append(historical_closes.iloc[-1] if not historical_closes.empty else np.nan)
            df_raw["Historical_Close"] = prices_at_quarter
        else:
            df_raw["Historical_Close"] = np.nan

        df_raw["P_E_TTM"] = safe_divide(df_raw["Historical_Close"], df_raw["TTM_EPS"]).where(df_raw["TTM_EPS"] > 0).clip(0, 250)
        implied_hist_cap = df_raw["Historical_Close"] * (df_raw["Diluted_Shares_M"] * 1e6)
        df_raw["P_S_TTM"] = safe_divide(implied_hist_cap, df_raw["TTM_Revenue"]).clip(0, 150)
        df_raw["FCF_Yield_%"] = (safe_divide(df_raw["TTM_FCF"], implied_hist_cap) * 100).clip(-50, 100)

        # ----------------------------------------------------
        # DASHBOARD HEADER
        # ----------------------------------------------------
        st.subheader(f"{ticker_symbol} — Executive Financial Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${current_price:,.2f}" if pd.notna(current_price) else "N/A")
        c2.metric("Market Cap", f"${market_cap / 1e9:,.2f}B" if pd.notna(market_cap) else "N/A")
        c3.metric("XBRL Quarters Extracted", str(df_all["Source"].astype(str).str.contains("SEC").sum()))
        c4.metric("Total Filings Analysed", str(len(df_all)))

        # ----------------------------------------------------
        # DIAGNOSTICS ACCORDION
        # ----------------------------------------------------
        with st.expander("Data Source Lineage & Diagnostics", expanded=False):
            diag_cols = ["Period", "Source"] + [
                c for c in df_all.columns if c.endswith("_Concept") or c.endswith("_Method")
            ]
            st.dataframe(df_all[[c for c in diag_cols if c in df_all.columns]].tail(20), use_container_width=True)

            if fact_meta:
                concept_rows = [
                    {"Metric": k, "Taxonomy": v["taxonomy"], "Concept": v["concept"], "Unit": v["unit"]}
                    for k, v in fact_meta.items() if v
                ]
                st.dataframe(pd.DataFrame(concept_rows), use_container_width=True)

            if diagnostics:
                for msg in diagnostics[-10:]:
                    st.caption(msg)

        # ----------------------------------------------------
        # RAW DATA ACCORDION
        # ----------------------------------------------------
        with st.expander("Quarterly Financial Dataset", expanded=False):
            display = df_raw.copy()
            display["Period"] = display["Period"].dt.strftime("%Y-%m-%d")
            display_cols = [
                "Period", "Source", "Revenue_B", "Rev_YoY_%", "Diluted_EPS", "EPS_YoY_%",
                "Operating_Income", "Op_Margin_%", "Net_Income", "Net_Margin_%",
                "OCF", "Capex", "FCF_B", "Diluted_Shares_M", "Share_Dilution_YoY_%",
            ]
            st.dataframe(display[[c for c in display_cols if c in display.columns]], use_container_width=True)

        # ----------------------------------------------------
        # CHARTS: FINANCIAL PERFORMANCE
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Financial Performance")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
        fig.suptitle(f"{ticker_symbol} Operational & Flow Fundamentals", fontsize=14, fontweight="bold", y=0.98)
        xlabels = df_raw["Period"].dt.strftime("%Y-%m-%d")

        # 1. Revenue
        if df_raw["Revenue_B"].notna().any():
            ax1.bar(xlabels, df_raw["Revenue_B"], width=0.55, alpha=0.85, label="Revenue ($B)", color="#1f77b4")
            ax1.set_title("Revenue ($B) & YoY Growth", fontweight="bold", fontsize=10.5)
            ax1.set_ylabel("Revenue ($B)")
            ax1.tick_params(axis="x", rotation=45, labelsize=7)

            ax1_sub = ax1.twinx()
            yoy_rev = df_raw["Rev_YoY_%"].dropna()
            if not yoy_rev.empty:
                ax1_sub.plot(xlabels[yoy_rev.index], yoy_rev, marker="o", color="#ff7f0e", linewidth=1.5, label="YoY Growth (%)")
            ax1_sub.set_ylabel("Growth (%)")
            ax1_sub.grid(False)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax1_sub.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

        # 2. EPS
        if df_raw["Diluted_EPS"].notna().any():
            ax2.bar(xlabels, df_raw["Diluted_EPS"], width=0.55, alpha=0.85, label="Diluted EPS ($)", color="#2ca02c")
            ax2.set_title("Diluted EPS ($) & YoY Growth", fontweight="bold", fontsize=10.5)
            ax2.set_ylabel("EPS ($)")
            ax2.tick_params(axis="x", rotation=45, labelsize=7)

            ax2_sub = ax2.twinx()
            yoy_eps = df_raw["EPS_YoY_%"].dropna()
            if not yoy_eps.empty:
                ax2_sub.plot(xlabels[yoy_eps.index], yoy_eps, marker="o", color="#d62728", linewidth=1.5, label="YoY Growth (%)")
            ax2_sub.set_ylabel("Growth (%)")
            ax2_sub.grid(False)

            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2_sub.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

        # 3. FCF
        if df_raw["FCF_B"].notna().any():
            ax3.bar(xlabels, df_raw["FCF_B"], width=0.55, alpha=0.85, label="Free Cash Flow ($B)", color="#9467bd")
            ax3.set_title("Standalone Free Cash Flow ($B) & YoY Growth", fontweight="bold", fontsize=10.5)
            ax3.set_ylabel("FCF ($B)")
            ax3.tick_params(axis="x", rotation=45, labelsize=7)

            ax3_sub = ax3.twinx()
            yoy_fcf = df_raw["FCF_YoY_%"].dropna()
            if not yoy_fcf.empty:
                ax3_sub.plot(xlabels[yoy_fcf.index], yoy_fcf, marker="o", color="#8c564b", linewidth=1.5, label="YoY Growth (%)")
            ax3_sub.set_ylabel("Growth (%)")
            ax3_sub.grid(False)

            h1, l1 = ax3.get_legend_handles_labels()
            h2, l2 = ax3_sub.get_legend_handles_labels()
            ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

        # 4. Margins
        valid_op = df_raw["Op_Margin_%"].dropna()
        valid_net = df_raw["Net_Margin_%"].dropna()
        if not valid_op.empty:
            ax4.plot(xlabels[valid_op.index], valid_op, marker="o", linewidth=2, label="Operating Margin (%)", color="#1f77b4")
        if not valid_net.empty:
            ax4.plot(xlabels[valid_net.index], valid_net, marker="s", linestyle="--", linewidth=2, label="Net Margin (%)", color="#2ca02c")

        ax4.axhline(0, linestyle=":", linewidth=1, alpha=0.6, color="gray")
        ax4.set_title("Operating Margin vs Net Margin (%)", fontweight="bold", fontsize=10.5)
        ax4.set_ylabel("Margin (%)")
        ax4.tick_params(axis="x", rotation=45, labelsize=7)
        ax4.legend(loc="upper left", fontsize=7)
        ax4.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # CHARTS: HISTORICAL VALUATION & PRICE
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Historical Valuation & Price Action")

        fig2, ((ax_p1, ax_p2), (ax_p3, ax_p4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
        fig2.suptitle(f"{ticker_symbol} As-Of Period Valuation (Non-Distorted)", fontsize=14, fontweight="bold", y=0.98)

        # 1. Daily Price & EMAs (Last 2 Years)
        if hist_price is not None and not hist_price.empty:
            two_years_ago = hist_price.index.max() - pd.DateOffset(years=2)
            recent_hist = hist_price[hist_price.index >= two_years_ago]
            ax_p1.plot(recent_hist.index, recent_hist["Close"], linewidth=1.5, label="Close Price ($)", color="black")
            ax_p1.plot(recent_hist.index, recent_hist["EMA50"], linewidth=1.2, label="50-Day EMA", color="#1f77b4")
            ax_p1.plot(recent_hist.index, recent_hist["EMA200"], linestyle="--", linewidth=1.2, label="200-Day EMA", color="#d62728")

        ax_p1.set_title("Daily Stock Price vs 50/200 EMA (2-Year)", fontweight="bold", fontsize=10.5)
        ax_p1.set_ylabel("Price ($)")
        ax_p1.legend(loc="upper left", fontsize=7)
        ax_p1.grid(True, linestyle="--", alpha=0.3)

        # 2. Historical P/S TTM
        valid_ps = df_raw["P_S_TTM"].dropna()
        if not valid_ps.empty:
            ax_p2.plot(xlabels[valid_ps.index], valid_ps, marker="o", linewidth=2, color="#17becf", label="Historical P/S (TTM)")
        ax_p2.set_title("Historical Price-to-Sales (TTM as-of Quarter)", fontweight="bold", fontsize=10.5)
        ax_p2.set_ylabel("P/S Multiple (x)")
        ax_p2.tick_params(axis="x", rotation=45, labelsize=7)
        ax_p2.legend(loc="upper left", fontsize=7)
        ax_p2.grid(True, linestyle="--", alpha=0.3)

        # 3. Historical P/E TTM
        valid_pe = df_raw["P_E_TTM"].dropna()
        if not valid_pe.empty:
            ax_p3.plot(xlabels[valid_pe.index], valid_pe, marker="s", linewidth=2, color="#bcbd22", label="Historical P/E (TTM)")
            ax_p3.axhline(0, linestyle=":", linewidth=1, alpha=0.6, color="gray")
        ax_p3.set_title("Historical Price-to-Earnings (TTM as-of Quarter)", fontweight="bold", fontsize=10.5)
        ax_p3.set_ylabel("P/E Multiple (x)")
        ax_p3.tick_params(axis="x", rotation=45, labelsize=7)
        ax_p3.legend(loc="upper left", fontsize=7)
        ax_p3.grid(True, linestyle="--", alpha=0.3)

        # 4. Historical FCF Yield TTM
        valid_fcfy = df_raw["FCF_Yield_%"].dropna()
        if not valid_fcfy.empty:
            ax_p4.plot(xlabels[valid_fcfy.index], valid_fcfy, marker="^", linewidth=2, color="#7f7f7f", label="Historical FCF Yield (%)")
            ax_p4.axhline(0, linestyle=":", linewidth=1, alpha=0.6, color="gray")
        ax_p4.set_title("Historical FCF Yield (TTM as-of Quarter)", fontweight="bold", fontsize=10.5)
        ax_p4.set_ylabel("FCF Yield (%)")
        ax_p4.tick_params(axis="x", rotation=45, labelsize=7)
        ax_p4.legend(loc="upper left", fontsize=7)
        ax_p4.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ----------------------------------------------------
        # CHARTS: CAPITAL STRUCTURE — SHARES & CAPEX (re-added)
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Capital Structure & Investment")

        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)
        fig3.suptitle(f"{ticker_symbol} Share Count & Capital Expenditure", fontsize=14, fontweight="bold", y=1.03)

        # 5. Diluted Shares Outstanding & YoY Dilution
        if df_raw["Diluted_Shares_M"].notna().any():
            ax5.bar(xlabels, df_raw["Diluted_Shares_M"], width=0.55, alpha=0.85, label="Diluted Shares (M)", color="#e377c2")
            ax5.set_title("Diluted Shares Outstanding (M) & YoY Dilution", fontweight="bold", fontsize=10.5)
            ax5.set_ylabel("Shares (Millions)")
            ax5.tick_params(axis="x", rotation=45, labelsize=7)

            ax5_sub = ax5.twinx()
            yoy_dil = df_raw["Share_Dilution_YoY_%"].dropna()
            if not yoy_dil.empty:
                ax5_sub.plot(xlabels[yoy_dil.index], yoy_dil, marker="o", color="#17becf", linewidth=1.5, label="YoY Dilution (%)")
            ax5_sub.set_ylabel("Dilution (%)")
            ax5_sub.grid(False)

            h1, l1 = ax5.get_legend_handles_labels()
            h2, l2 = ax5_sub.get_legend_handles_labels()
            ax5.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

        # 6. Capex & YoY Growth
        if df_raw["Capex_Positive"].notna().any():
            capex_m = df_raw["Capex_Positive"] / 1e6
            ax6.bar(xlabels, capex_m, width=0.55, alpha=0.85, label="Capex ($M)", color="#8c564b")
            ax6.set_title("Capital Expenditure ($M) & YoY Growth", fontweight="bold", fontsize=10.5)
            ax6.set_ylabel("Capex ($M)")
            ax6.tick_params(axis="x", rotation=45, labelsize=7)

            ax6_sub = ax6.twinx()
            capex_yoy = pct_change_safe(df_raw["Capex_Positive"], 4).clip(-500, 500).dropna()
            if not capex_yoy.empty:
                ax6_sub.plot(xlabels[capex_yoy.index], capex_yoy, marker="o", color="#bcbd22", linewidth=1.5, label="YoY Growth (%)")
            ax6_sub.set_ylabel("Growth (%)")
            ax6_sub.grid(False)

            h1, l1 = ax6.get_legend_handles_labels()
            h2, l2 = ax6_sub.get_legend_handles_labels()
            ax6.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as exc:
        st.error(f"Could not load data for ticker '{ticker_symbol}'. Error: {type(exc).__name__}: {exc}")
        st.exception(exc)
