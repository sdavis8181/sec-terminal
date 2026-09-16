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
    # --- Added for ROIC: pretax income & tax expense are duration (flow)
    # facts just like Revenue/OCF, so they ride the exact same extraction,
    # Q4-derivation, canonicalization and reindexing pipeline as everything else.
    "Pretax_Income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "Income_Tax_Expense": [
        "IncomeTaxExpenseBenefit",
        "IncomeTaxExpenseBenefitContinuingOperations",
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
    "Pretax_Income",
    "Income_Tax_Expense",
]

# Balance-sheet (instant, point-in-time) concepts used for Invested Capital.
# These are NOT duration facts, so they need their own lightweight extraction
# path (see get_instant_entries / build_balance_sheet_table below) rather than
# the Q1/6M/9M/12M subtraction logic used for flow facts.
BALANCE_CONCEPTS = {
    "Total_Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Long_Term_Debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermNotesPayable",
    ],
    "Short_Term_Debt": [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "NotesPayableCurrent",
    ],
    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
    ],
}
BALANCE_FIELDS = ["Total_Equity", "Long_Term_Debt", "Short_Term_Debt", "Cash"]


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

    full_index = pd.date_range(start=df["Period"].min(), end=df["Period"].max(), freq="QE")
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
        if field in {"OCF", "Capex", "Revenue", "Operating_Income", "Net_Income", "Pretax_Income", "Income_Tax_Expense"}:
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


def get_instant_entries(companyfacts, field):
    """
    Like get_all_usable_entries, but for balance-sheet ("instant") facts,
    which only have an "end" date (a snapshot), never a "start"/duration.
    """
    facts = companyfacts.get("facts", {})
    preferred = BALANCE_CONCEPTS.get(field, [])
    all_entries = []

    for taxonomy, taxonomy_facts in facts.items():
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept in preferred:
            if concept not in taxonomy_facts:
                continue
            concept_data = taxonomy_facts[concept]
            unit = choose_unit_name(concept_data, field) or "USD"
            entries = concept_data.get("units", {}).get(unit, [])
            for e in entries:
                if isinstance(e, dict) and "val" in e and "end" in e and "start" not in e:
                    all_entries.append(dict(e, taxonomy=taxonomy, concept=concept, unit=unit))

    return all_entries


def build_balance_sheet_table(companyfacts):
    """
    Builds a quarter-indexed table of instant balance-sheet items (equity,
    debt, cash) used to compute Invested Capital for the ROIC chart.
    """
    rows = {}
    meta = {}

    for field in BALANCE_FIELDS:
        entries = get_instant_entries(companyfacts, field)
        if not entries:
            meta[field] = None
            continue

        meta[field] = {
            "taxonomy": entries[0]["taxonomy"],
            "concept": entries[0]["concept"],
            "unit": entries[0]["unit"],
        }

        grouped = {}
        for e in entries:
            end = entry_date(e, "end")
            if pd.isna(end):
                continue
            key = canonical_quarter_end(end).strftime("%Y-%m-%d")
            grouped.setdefault(key, []).append(e)

        for key, group in grouped.items():
            best = sorted(group, key=lambda x: (str(x.get("filed", "")), str(x.get("accn", ""))))[-1]
            canon_period = canonical_quarter_end(entry_date(best, "end"))
            rows.setdefault(key, {"Period": canon_period})
            rows[key][field] = clean_number(best.get("val"))
            rows[key][f"{field}_Concept"] = f"{best.get('taxonomy')}:{best.get('concept')}"

    df = pd.DataFrame(list(rows.values()))
    return df, meta


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

    # Invested Capital components (instant balance-sheet facts merged in by
    # fetch_and_parse_ticker). Kept as plain per-quarter snapshots here; the
    # TTM/NOPAT/ROIC math happens later in main() alongside the other
    # TTM-dependent valuation metrics (P/E, P/S, FCF yield), for consistency.
    for col in BALANCE_FIELDS:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["Total_Debt"] = work[["Long_Term_Debt", "Short_Term_Debt"]].sum(axis=1, min_count=1)
    work["Invested_Capital"] = work["Total_Debt"].fillna(0) + work["Total_Equity"] - work["Cash"].fillna(0)

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

    # Merge in instant balance-sheet facts (equity/debt/cash) for ROIC. Both
    # sides are already on the canonical quarter-end grid, so this is a
    # straightforward left-merge on Period.
    balance_df, balance_meta = build_balance_sheet_table(companyfacts)
    if not balance_df.empty:
        df = df.merge(balance_df, on="Period", how="left")
    for field, info in balance_meta.items():
        fact_meta[field] = info
    if all(v is None for v in balance_meta.values()):
        diagnostics.append("No balance-sheet (equity/debt/cash) facts found - ROIC will be unavailable.")

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
        # ROIC (Return on Invested Capital)
        # NOPAT (TTM) = TTM Operating Income x (1 - effective tax rate)
        # Effective tax rate = TTM Income Tax Expense / TTM Pretax Income,
        # falling back to a flat 21% statutory rate when tax facts aren't
        # available or pretax income is <= 0 (rate would be meaningless).
        # Invested Capital = Total Debt + Total Equity - Cash, averaged
        # between the start and end of the trailing-twelve-month window.
        # ----------------------------------------------------
        df_raw["TTM_Operating_Income"] = calculate_trailing_flow_strict(df_raw["Operating_Income"])
        df_raw["TTM_Pretax_Income"] = calculate_trailing_flow_strict(df_raw["Pretax_Income"])
        df_raw["TTM_Income_Tax_Expense"] = calculate_trailing_flow_strict(df_raw["Income_Tax_Expense"])

        eff_tax_rate = safe_divide(df_raw["TTM_Income_Tax_Expense"], df_raw["TTM_Pretax_Income"])
        eff_tax_rate = eff_tax_rate.where(df_raw["TTM_Pretax_Income"] > 0).clip(0, 0.6)
        eff_tax_rate = eff_tax_rate.fillna(0.21)

        df_raw["NOPAT_TTM"] = df_raw["TTM_Operating_Income"] * (1 - eff_tax_rate)

        df_raw["Invested_Capital_Avg"] = (df_raw["Invested_Capital"] + df_raw["Invested_Capital"].shift(4)) / 2
        df_raw["Invested_Capital_Avg"] = df_raw["Invested_Capital_Avg"].fillna(df_raw["Invested_Capital"])

        df_raw["ROIC_%"] = safe_divide(df_raw["NOPAT_TTM"], df_raw["Invested_Capital_Avg"]) * 100
        df_raw["ROIC_%"] = df_raw["ROIC_%"].where(df_raw["Invested_Capital_Avg"] > 0).clip(-100, 100)

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
                "Total_Equity", "Total_Debt", "Cash", "Invested_Capital",
            ]
            st.dataframe(display[[c for c in display_cols if c in display.columns]], use_container_width=True)

        # ----------------------------------------------------
        # Shared helper for the "bar + secondary-axis line" chart style
        # used throughout every row below.
        # ----------------------------------------------------
        xlabels = df_raw["Period"].dt.strftime("%Y-%m-%d")

        def bar_with_growth(ax, values, growth, bar_label, growth_label, bar_color, line_color, title, ylabel):
            if values.notna().any():
                ax.bar(xlabels, values, width=0.55, alpha=0.85, label=bar_label, color=bar_color)
                ax.set_title(title, fontweight="bold", fontsize=10.5)
                ax.set_ylabel(ylabel)
                ax.tick_params(axis="x", rotation=45, labelsize=7)

                ax_sub = ax.twinx()
                g = growth.dropna()
                if not g.empty:
                    ax_sub.plot(xlabels[g.index], g, marker="o", color=line_color, linewidth=1.5, label=growth_label)
                ax_sub.set_ylabel("Growth (%)")
                ax_sub.grid(False)

                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax_sub.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
            else:
                ax.set_title(title, fontweight="bold", fontsize=10.5)
                ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")

        def line_chart(ax, values, label, color, title, ylabel, marker="o", zero_line=False):
            v = values.dropna()
            if not v.empty:
                ax.plot(xlabels[v.index], v, marker=marker, linewidth=2, color=color, label=label)
                if zero_line:
                    ax.axhline(0, linestyle=":", linewidth=1, alpha=0.6, color="gray")
                ax.legend(loc="upper left", fontsize=7)
            else:
                ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(title, fontweight="bold", fontsize=10.5)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)

        st.markdown("---")

        # ======================================================
        # ROW 1 — Stock Price (EMAs) & Margins
        # ======================================================
        st.subheader(f"{ticker_symbol} — Price Action & Margins")
        fig1, (ax_price, ax_margin) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

        if hist_price is not None and not hist_price.empty:
            two_years_ago = hist_price.index.max() - pd.DateOffset(years=2)
            recent_hist = hist_price[hist_price.index >= two_years_ago]
            ax_price.plot(recent_hist.index, recent_hist["Close"], linewidth=1.5, label="Close Price ($)", color="black")
            ax_price.plot(recent_hist.index, recent_hist["EMA50"], linewidth=1.2, label="50-Day EMA", color="#1f77b4")
            ax_price.plot(recent_hist.index, recent_hist["EMA200"], linestyle="--", linewidth=1.2, label="200-Day EMA", color="#d62728")
            ax_price.legend(loc="upper left", fontsize=7)
        else:
            ax_price.text(0.5, 0.5, "No price data available", ha="center", va="center", transform=ax_price.transAxes, fontsize=9, color="gray")
        ax_price.set_title("Daily Stock Price vs 50/200 EMA (2-Year)", fontweight="bold", fontsize=10.5)
        ax_price.set_ylabel("Price ($)")
        ax_price.grid(True, linestyle="--", alpha=0.3)

        valid_op = df_raw["Op_Margin_%"].dropna()
        valid_net = df_raw["Net_Margin_%"].dropna()
        if not valid_op.empty:
            ax_margin.plot(xlabels[valid_op.index], valid_op, marker="o", linewidth=2, label="Operating Margin (%)", color="#1f77b4")
        if not valid_net.empty:
            ax_margin.plot(xlabels[valid_net.index], valid_net, marker="s", linestyle="--", linewidth=2, label="Net Margin (%)", color="#2ca02c")
        if valid_op.empty and valid_net.empty:
            ax_margin.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax_margin.transAxes, fontsize=9, color="gray")
        ax_margin.axhline(0, linestyle=":", linewidth=1, alpha=0.6, color="gray")
        ax_margin.set_title("Operating Margin vs Net Margin (%)", fontweight="bold", fontsize=10.5)
        ax_margin.set_ylabel("Margin (%)")
        ax_margin.tick_params(axis="x", rotation=45, labelsize=7)
        ax_margin.legend(loc="upper left", fontsize=7)
        ax_margin.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ======================================================
        # ROW 2 — Revenue & Earnings Growth
        # ======================================================
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Revenue & Earnings Growth")
        fig2, (ax_rev, ax_eps) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

        bar_with_growth(ax_rev, df_raw["Revenue_B"], df_raw["Rev_YoY_%"], "Revenue ($B)", "YoY Growth (%)",
                         "#1f77b4", "#ff7f0e", "Revenue ($B) & YoY Growth", "Revenue ($B)")
        bar_with_growth(ax_eps, df_raw["Diluted_EPS"], df_raw["EPS_YoY_%"], "Diluted EPS ($)", "YoY Growth (%)",
                         "#2ca02c", "#d62728", "Diluted EPS ($) & YoY Growth", "EPS ($)")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ======================================================
        # ROW 3 — P/S & P/E
        # ======================================================
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Valuation Multiples")
        fig3, (ax_ps, ax_pe) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

        line_chart(ax_ps, df_raw["P_S_TTM"], "Historical P/S (TTM)", "#17becf",
                   "Historical Price-to-Sales (TTM as-of Quarter)", "P/S Multiple (x)")
        line_chart(ax_pe, df_raw["P_E_TTM"], "Historical P/E (TTM)", "#bcbd22",
                   "Historical Price-to-Earnings (TTM as-of Quarter)", "P/E Multiple (x)",
                   marker="s", zero_line=True)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ======================================================
        # ROW 4 — FCF Growth & FCF Yield
        # ======================================================
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Free Cash Flow")
        fig4, (ax_fcf, ax_fcfy) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

        bar_with_growth(ax_fcf, df_raw["FCF_B"], df_raw["FCF_YoY_%"], "Free Cash Flow ($B)", "YoY Growth (%)",
                         "#9467bd", "#8c564b", "Standalone Free Cash Flow ($B) & YoY Growth", "FCF ($B)")
        line_chart(ax_fcfy, df_raw["FCF_Yield_%"], "Historical FCF Yield (%)", "#7f7f7f",
                   "Historical FCF Yield (TTM as-of Quarter)", "FCF Yield (%)",
                   marker="^", zero_line=True)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # ======================================================
        # ROW 5 — Capex & Shares Outstanding
        # ======================================================
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Capital Structure & Investment")
        fig5, (ax_capex, ax_shares) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

        capex_m = df_raw["Capex_Positive"] / 1e6
        capex_yoy = pct_change_safe(df_raw["Capex_Positive"], 4).clip(-500, 500)
        bar_with_growth(ax_capex, capex_m, capex_yoy, "Capex ($M)", "YoY Growth (%)",
                         "#8c564b", "#bcbd22", "Capital Expenditure ($M) & YoY Growth", "Capex ($M)")
        bar_with_growth(ax_shares, df_raw["Diluted_Shares_M"], df_raw["Share_Dilution_YoY_%"],
                         "Diluted Shares (M)", "YoY Dilution (%)", "#e377c2", "#17becf",
                         "Diluted Shares Outstanding (M) & YoY Dilution", "Shares (Millions)")

        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        # ======================================================
        # ROW 6 — ROIC
        # ======================================================
        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Return on Invested Capital")
        if df_raw["Invested_Capital"].isna().all():
            st.info(
                "ROIC unavailable: SEC XBRL company facts for this ticker don't include the "
                "balance-sheet tags (StockholdersEquity / debt / cash) this app looks for. "
                "See the diagnostics panel above for exact concept coverage."
            )

        fig6, (ax_roic, ax_nopat) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

        line_chart(ax_roic, df_raw["ROIC_%"], "ROIC, TTM (%)", "#2ca02c",
                   "Return on Invested Capital (TTM as-of Quarter)", "ROIC (%)",
                   marker="o", zero_line=True)

        nopat_m = df_raw["NOPAT_TTM"] / 1e6
        ic_m = df_raw["Invested_Capital_Avg"] / 1e6
        if nopat_m.notna().any() or ic_m.notna().any():
            ax_nopat.bar(xlabels, nopat_m, width=0.55, alpha=0.85, label="NOPAT, TTM ($M)", color="#1f77b4")
            ax_nopat.set_title("NOPAT (TTM) vs Avg. Invested Capital", fontweight="bold", fontsize=10.5)
            ax_nopat.set_ylabel("NOPAT ($M)")
            ax_nopat.tick_params(axis="x", rotation=45, labelsize=7)

            ax_nopat_sub = ax_nopat.twinx()
            ic_valid = ic_m.dropna()
            if not ic_valid.empty:
                ax_nopat_sub.plot(xlabels[ic_valid.index], ic_valid, marker="o", color="#d62728", linewidth=1.5, label="Avg. Invested Capital ($M)")
            ax_nopat_sub.set_ylabel("Invested Capital ($M)")
            ax_nopat_sub.grid(False)

            h1, l1 = ax_nopat.get_legend_handles_labels()
            h2, l2 = ax_nopat_sub.get_legend_handles_labels()
            ax_nopat.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
        else:
            ax_nopat.set_title("NOPAT (TTM) vs Avg. Invested Capital", fontweight="bold", fontsize=10.5)
            ax_nopat.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax_nopat.transAxes, fontsize=9, color="gray")

        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    except Exception as exc:
        st.error(f"Could not load data for ticker '{ticker_symbol}'. Error: {type(exc).__name__}: {exc}")
        st.exception(exc)
