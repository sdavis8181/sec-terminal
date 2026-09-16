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
    "Quarterly financial data from SEC XBRL (US-GAAP & IFRS), with Yahoo Finance "
    "used for market-price history and as a limited fallback."
)

DEFAULT_IDENTITY = "Scott Davis scott@example.com"
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
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Terminal Controls")

    ticker_symbol = (
        st.text_input("Stock Ticker", value="AXON", max_chars=12)
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
    run_button = st.button("Generate Report", type="primary")

    st.markdown("---")
    st.caption(
        "SEC XBRL is the primary financial-statement source. "
        "The parser uses individual XBRL facts and their actual reporting "
        "periods instead of taking the first year-looking table column."
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
    s = (
        s.replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
    )

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


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


# ============================================================
# SEC API
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_sec_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=TICKER_HEADERS, timeout=30)
    response.raise_for_status()

    raw = response.json()
    rows = []

    for item in raw.values():
        rows.append(
            {
                "ticker": str(item.get("ticker", "")).upper(),
                "title": item.get("title", ""),
                "cik": int(item.get("cik_str", 0)),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def get_cik_for_ticker(ticker):
    df = get_sec_ticker_map()

    match = df[df["ticker"].eq(ticker.upper())]

    if match.empty:
        raise ValueError(
            f"{ticker} was not found in the SEC ticker list."
        )

    return int(match.iloc[0]["cik"])


@st.cache_data(ttl=86400, show_spinner=False)
def get_companyfacts(cik):
    url = (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK"
        f"{int(cik):010d}.json"
    )

    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=45,
    )
    response.raise_for_status()

    return response.json()


# ============================================================
# XBRL FACT DEFINITIONS (US-GAAP + IFRS)
# ============================================================

FLOW_CONCEPTS = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractsWithCustomers",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
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
        "PaymentsToAcquirePropertyPlantAndEquipmentAndOtherPropertyPlantAndEquipment",
        "AdditionsToPropertyPlantAndEquipment",
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
# FACT UTILITIES
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
                candidates.append(
                    (
                        taxonomy,
                        concept,
                        taxonomy_facts[concept],
                    )
                )

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
        for unit in ["shares"]:
            if unit in units:
                return unit

    if field in {
        "Revenue",
        "Operating_Income",
        "Net_Income",
        "OCF",
        "Capex",
    }:
        for unit in ["USD", "EUR", "BRL", "ARS"]:
            if unit in units:
                return unit

        if len(units) == 1:
            return next(iter(units.keys()))

    if len(units) == 1:
        return next(iter(units.keys()))

    return None


def get_preferred_entries(companyfacts, field):
    candidates = all_fact_candidates(companyfacts, field)

    for taxonomy, concept, concept_data in candidates:
        unit = choose_unit_name(concept_data, field)

        if not unit:
            continue

        entries = concept_data["units"].get(unit, [])
        usable = [
            x for x in entries
            if isinstance(x, dict) and "val" in x
        ]

        if usable:
            return {
                "taxonomy": taxonomy,
                "concept": concept,
                "unit": unit,
                "entries": usable,
            }

    return None


def entry_date(entry, key):
    return as_date(entry.get(key))


def entry_duration_days(entry):
    start = entry_date(entry, "start")
    end = entry_date(entry, "end")

    if pd.isna(start) or pd.isna(end):
        return None

    return (end - start).days + 1


def entry_is_flow(entry):
    return "start" in entry and "end" in entry


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
        if not entry_is_flow(e):
            continue

        days = entry_duration_days(e)
        if days is None:
            continue

        form = str(e.get("form", "")).upper()
        frame = str(e.get("frame", ""))

        # Prioritize true filing duration end-date for off-calendar filers
        if 70 <= days <= 110:
            end = entry_date(e, "end")
            score = 60
            if form in {"10-Q", "20-F", "6-K"}:
                score += 20

            output.append(
                {
                    "entry": e,
                    "period": end,
                    "method": "duration",
                    "score": score,
                }
            )
            continue

        q_end = quarter_from_frame(frame)
        if q_end is not None:
            output.append(
                {
                    "entry": e,
                    "period": q_end,
                    "method": "SEC frame",
                    "score": 50,
                }
            )

    return output


def pick_best_entry(candidates):
    if not candidates:
        return None

    def sort_key(x):
        e = x["entry"] if "entry" in x else x
        filed = str(e.get("filed", ""))
        accession = str(e.get("accn", ""))
        return (filed, accession)

    return sorted(candidates, key=sort_key)[-1]


# ============================================================
# DEDUPLICATOR & SNAP ENGINE
# ============================================================

def collapse_duplicate_quarters(df, tolerance_days=45):
    """
    Collapses quarter entries within 45 days of each other (such as Yahoo's 03-31
    and Rubrik's fiscal 04-30) into one single period, giving full priority to
    SEC-sourced values over Yahoo values.
    """
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
            # Check proximity to any date already in the cluster
            if any(abs((p - existing["Period"]).days) <= tolerance_days for existing in cluster):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])

    unified_rows = []
    for cluster in clusters:
        # Prefer the SEC record for determining the canonical quarter-end date
        sec_rows = [r for r in cluster if "SEC" in str(r.get("Source", ""))]
        canonical_row = sec_rows[-1] if sec_rows else cluster[-1]
        canonical_period = canonical_row["Period"]

        merged_record = {"Period": canonical_period}
        
        # Combine non-null values, giving preference to SEC entries
        for col in df.columns:
            if col == "Period":
                continue
            val = np.nan
            # 1. Search SEC rows in cluster
            for r in sec_rows:
                if pd.notna(r.get(col)):
                    val = r[col]
            # 2. If still NaN, search Yahoo rows
            if pd.isna(val):
                for r in cluster:
                    if pd.notna(r.get(col)):
                        val = r[col]
            merged_record[col] = val

        # Set final combined source
        sources = {str(r.get("Source", "")) for r in cluster if pd.notna(r.get("Source"))}
        if any("SEC" in s for s in sources) and any("Yahoo" in s for s in sources):
            merged_record["Source"] = "SEC + Yahoo"
        elif any("SEC" in s for s in sources):
            merged_record["Source"] = "SEC XBRL"
        else:
            merged_record["Source"] = "Yahoo Finance"

        unified_rows.append(merged_record)

    out_df = pd.DataFrame(unified_rows).sort_values("Period").reset_index(drop=True)
    return out_df


# ============================================================
# BUILD QUARTERLY DATA
# ============================================================

def build_quarterly_fact_table(companyfacts):
    fact_meta = {}
    quarter_rows = {}

    for field in FLOW_FIELDS:
        meta = get_preferred_entries(companyfacts, field)

        if not meta:
            fact_meta[field] = None
            continue

        fact_meta[field] = meta

        entries = meta["entries"]
        candidates = standalone_quarter_candidates(entries)

        grouped = {}
        for item in candidates:
            period = item["period"]
            if pd.isna(period):
                continue
            key = period.strftime("%Y-%m-%d")
            grouped.setdefault(key, []).append(item)

        for key, group in grouped.items():
            best = pick_best_entry(group)
            if best is None:
                continue

            entry = best["entry"]
            quarter_rows.setdefault(
                key,
                {
                    "Period": best["period"],
                    "Source": "SEC XBRL",
                },
            )

            quarter_rows[key][field] = clean_number(entry.get("val"))
            quarter_rows[key][f"{field}_Concept"] = f"{meta['taxonomy']}:{meta['concept']}"
            quarter_rows[key][f"{field}_Method"] = best["method"]

    df = pd.DataFrame(list(quarter_rows.values()))

    if df.empty:
        return df, fact_meta

    df = collapse_duplicate_quarters(df, tolerance_days=45)
    df = df[df["Period"] >= pd.Timestamp("2017-01-01")].reset_index(drop=True)

    return df, fact_meta


# ============================================================
# YAHOO FALLBACK / SUPPLEMENT
# ============================================================

YF_KEYS = {
    "Revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "Operating_Income": ["Operating Income", "Operating Income Loss", "EBIT"],
    "Net_Income": ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"],
    "Diluted_EPS": ["Diluted EPS", "Diluted EPS From Continuing Operations"],
    "Diluted_Shares": ["Diluted Average Shares", "Diluted Average Shares Outstanding"],
    "OCF": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"],
    "Capex": ["Capital Expenditure", "Capital Expenditure Reported", "Purchase Of Property Plant And Equipment"],
}


def yahoo_row_value(df, keys, date_col):
    if df is None or df.empty:
        return np.nan

    for key in keys:
        if key in df.index:
            try:
                return clean_number(df.loc[key, date_col])
            except Exception:
                pass

    return np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_quarterly(ticker):
    diagnostics = []

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

            rows.append(
                {
                    "Period": period,
                    "Source": "Yahoo Finance",
                    "Revenue": yahoo_row_value(inc, YF_KEYS["Revenue"], date_col),
                    "Operating_Income": yahoo_row_value(inc, YF_KEYS["Operating_Income"], date_col),
                    "Net_Income": yahoo_row_value(inc, YF_KEYS["Net_Income"], date_col),
                    "Diluted_EPS": yahoo_row_value(inc, YF_KEYS["Diluted_EPS"], date_col),
                    "Diluted_Shares": yahoo_row_value(inc, YF_KEYS["Diluted_Shares"], date_col),
                    "OCF": yahoo_row_value(cf, YF_KEYS["OCF"], date_col),
                    "Capex": yahoo_row_value(cf, YF_KEYS["Capex"], date_col),
                }
            )

        return pd.DataFrame(rows), diagnostics

    except Exception as exc:
        return pd.DataFrame(), [f"Yahoo error: {type(exc).__name__}: {exc}"]


def merge_sec_yahoo(sec_df, yahoo_df):
    if sec_df.empty:
        return yahoo_df.copy()
    if yahoo_df.empty:
        return sec_df.copy()

    # Combine both datasets and apply the cluster-snapping deduplicator
    combined = pd.concat([sec_df, yahoo_df], ignore_index=True)
    return collapse_duplicate_quarters(combined, tolerance_days=45)


# ============================================================
# FINANCIAL DERIVATIONS
# ============================================================

def calculate_metrics(df):
    work = df.copy()

    numeric_cols = [
        "Revenue", "Operating_Income", "Net_Income",
        "Diluted_EPS", "Diluted_Shares", "OCF", "Capex",
    ]

    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["Revenue_B"] = work["Revenue"] / 1e9
    work["Rev_YoY_%"] = pct_change_safe(work["Revenue_B"], 4)
    work["Rev_QoQ_%"] = pct_change_safe(work["Revenue_B"], 1)

    work["EPS_YoY_%"] = pct_change_safe(work["Diluted_EPS"], 4).clip(-500, 500)
    work["EPS_QoQ_%"] = pct_change_safe(work["Diluted_EPS"], 1).clip(-500, 500)

    work["Op_Margin_%"] = safe_divide(work["Operating_Income"], work["Revenue"]) * 100
    work["Net_Margin_%"] = safe_divide(work["Net_Income"], work["Revenue"]) * 100

    work["Diluted_Shares_M"] = work["Diluted_Shares"] / 1e6
    work["Share_Dilution_YoY_%"] = pct_change_safe(work["Diluted_Shares_M"], 4).clip(-50, 50)

    return work


def calculate_fcf(df):
    work = df[["Period", "OCF", "Capex", "Source"]].copy()

    work["OCF"] = pd.to_numeric(work["OCF"], errors="coerce")
    work["Capex"] = pd.to_numeric(work["Capex"], errors="coerce")
    work["Capex_Positive"] = work["Capex"].abs()

    work["FCF"] = work["OCF"] - work["Capex_Positive"]
    work["FCF_B"] = work["FCF"] / 1e9
    work["FCF_YoY_%"] = pct_change_safe(work["FCF"], 4).clip(-500, 500)
    work["FCF_QoQ_%"] = pct_change_safe(work["FCF"], 1).clip(-500, 500)

    return work


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            pass

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        shares_outstanding = info.get("sharesOutstanding")
        market_cap = info.get("marketCap")

        hist = tk.history(period="2y", auto_adjust=False)

        if current_price is None and hist is not None and not hist.empty:
            current_price = clean_number(hist["Close"].iloc[-1])

        if market_cap is None and current_price is not None and shares_outstanding is not None:
            market_cap = current_price * shares_outstanding

        if hist is not None and not hist.empty:
            hist = hist.copy()
            hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()

        return (
            hist,
            clean_number(current_price),
            clean_number(market_cap),
            clean_number(shares_outstanding),
        )

    except Exception:
        return pd.DataFrame(), np.nan, np.nan, np.nan


# ============================================================
# SEC DATA PIPELINE
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

    df = merge_sec_yahoo(sec_df, yahoo_df)

    if df.empty:
        raise ValueError(f"No usable quarterly data found for {ticker}.")

    df = calculate_metrics(df)

    implied = safe_divide(df["Net_Income"], df["Diluted_EPS"])
    missing = (
        df["Diluted_Shares"].isna()
        & implied.notna()
        & (df["Diluted_EPS"].abs() > 0.0001)
    )

    df.loc[missing, "Diluted_Shares"] = implied[missing]
    df.loc[missing, "Diluted_Shares_Method"] = "Implied Net Income / Diluted EPS"
    df["Diluted_Shares_M"] = df["Diluted_Shares"] / 1e6
    df["Share_Dilution_YoY_%"] = pct_change_safe(df["Diluted_Shares_M"], 4).clip(-50, 50)

    return df, fact_meta, diagnostics


# ============================================================
# RUN
# ============================================================

if run_button or ticker_symbol:
    try:
        # Nonce flips on force refresh to guarantee cache bust
        nonce = np.random.randint(1, 1000000) if force_refresh else 0

        with st.spinner(f"Loading SEC XBRL and market data for {ticker_symbol}..."):
            df_all, fact_meta, diagnostics = fetch_and_parse_ticker(ticker_symbol, refresh_nonce=nonce)
            hist_price, current_price, market_cap, shares_outstanding = fetch_market_data(ticker_symbol)

        df_raw = df_all.tail(lookback_quarters).reset_index(drop=True)
        df_fcf = calculate_fcf(df_raw)

        # ----------------------------------------------------
        # TTM VALUATION
        # ----------------------------------------------------

        df_raw["TTM_Revenue"] = df_raw["Revenue"].rolling(4).sum()
        df_raw["TTM_EPS"] = df_raw["Diluted_EPS"].rolling(4).sum()
        df_fcf["TTM_FCF"] = df_fcf["FCF"].rolling(4).sum()

        if pd.notna(market_cap) and market_cap > 0 and pd.notna(current_price) and current_price > 0:
            df_raw["P_S_TTM"] = safe_divide(market_cap, df_raw["TTM_Revenue"]).clip(lower=0, upper=150)
            df_raw["P_E_TTM"] = safe_divide(current_price, df_raw["TTM_EPS"]).where(df_raw["TTM_EPS"] > 0).clip(lower=0, upper=250)
            df_raw["TTM_FCF"] = df_fcf["TTM_FCF"].values
            df_raw["FCF_Yield_%"] = (df_raw["TTM_FCF"] / market_cap) * 100

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        st.subheader(f"{ticker_symbol} — Executive Financial Dashboard")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Current Price", f"${current_price:,.2f}" if pd.notna(current_price) else "N/A")
        with c2:
            st.metric("Market Cap", f"${market_cap / 1e9:,.2f}B" if pd.notna(market_cap) else "N/A")
        with c3:
            st.metric("SEC / XBRL Quarters", str(df_all["Source"].astype(str).str.contains("SEC").sum()))
        with c4:
            st.metric("Quarterly Records", str(len(df_all)))

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        with st.expander("Data Source Diagnostics", expanded=False):
            st.write("**Quarterly source by period:**")
            diag_cols = ["Period", "Source"]
            for col in [
                "Revenue_Concept", "Revenue_Method", "Diluted_EPS_Concept", "Diluted_EPS_Method",
                "Diluted_Shares_Concept", "Diluted_Shares_Method", "OCF_Concept", "OCF_Method",
                "Capex_Concept", "Capex_Method",
            ]:
                if col in df_all.columns:
                    diag_cols.append(col)

            st.dataframe(df_all[diag_cols].tail(20), use_container_width=True)

            if fact_meta:
                st.write("**XBRL concepts selected:**")
                concept_rows = []
                for field, meta in fact_meta.items():
                    if meta:
                        concept_rows.append(
                            {
                                "Metric": field,
                                "Taxonomy": meta["taxonomy"],
                                "Concept": meta["concept"],
                                "Unit": meta["unit"],
                            }
                        )
                if concept_rows:
                    st.dataframe(pd.DataFrame(concept_rows), use_container_width=True)

            if diagnostics:
                st.write("**Diagnostics:**")
                for message in diagnostics[-20:]:
                    st.caption(message)

        # ----------------------------------------------------
        # RAW TABLE
        # ----------------------------------------------------

        with st.expander("Quarterly Financial Dataset", expanded=False):
            display = df_raw.copy()
            display["Period"] = display["Period"].dt.strftime("%Y-%m-%d")

            display_cols = [
                "Period", "Source", "Revenue_B", "Rev_YoY_%", "Diluted_EPS", "EPS_YoY_%",
                "Operating_Income", "Op_Margin_%", "Net_Income", "Net_Margin_%",
                "OCF", "Capex", "Diluted_Shares_M", "Share_Dilution_YoY_%",
            ]
            display_cols = [c for c in display_cols if c in display.columns]
            st.dataframe(display[display_cols], use_container_width=True)

        # ----------------------------------------------------
        # CHART 1: FINANCIALS
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Financial Performance")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
        fig.suptitle(f"{ticker_symbol} Financial & Growth Dashboard", fontsize=15, fontweight="bold", y=0.98)

        xlabels = df_raw["Period"].dt.strftime("%Y-%m-%d")

        # Revenue
        valid_rev = df_raw["Revenue_B"].notna()
        if valid_rev.any():
            ax1.bar(xlabels[valid_rev], df_raw.loc[valid_rev, "Revenue_B"], width=0.55, alpha=0.85, label="Revenue ($B)")
            ax1.set_title("Revenue ($B) & Growth", fontweight="bold", fontsize=10.5)
            ax1.set_ylabel("Revenue ($B)")
            ax1.tick_params(axis="x", rotation=45, labelsize=7)

            ax1_sub = ax1.twinx()
            ax1_sub.plot(xlabels[valid_rev], df_raw.loc[valid_rev, "Rev_YoY_%"], marker="o", linewidth=1.5, label="YoY Growth (%)")
            ax1_sub.plot(xlabels[valid_rev], df_raw.loc[valid_rev, "Rev_QoQ_%"], marker="s", linestyle="--", linewidth=1.2, label="QoQ Growth (%)")
            ax1_sub.set_ylabel("Growth (%)")
            ax1_sub.grid(False)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax1_sub.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.5)

        # EPS
        valid_eps = df_raw["Diluted_EPS"].notna()
        if valid_eps.any():
            ax2.bar(xlabels[valid_eps], df_raw.loc[valid_eps, "Diluted_EPS"], width=0.55, alpha=0.85, label="Diluted EPS ($)")
            ax2.set_title("Diluted EPS ($) & Growth", fontweight="bold", fontsize=10.5)
            ax2.set_ylabel("EPS ($)")
            ax2.tick_params(axis="x", rotation=45, labelsize=7)

            ax2_sub = ax2.twinx()
            ax2_sub.plot(xlabels[valid_eps], df_raw.loc[valid_eps, "EPS_YoY_%"], marker="o", linewidth=1.5, label="YoY Growth (%)")
            ax2_sub.plot(xlabels[valid_eps], df_raw.loc[valid_eps, "EPS_QoQ_%"], marker="s", linestyle="--", linewidth=1.2, label="QoQ Growth (%)")
            ax2_sub.set_ylabel("Growth (%)")
            ax2_sub.grid(False)

            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2_sub.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.5)

        # FCF
        valid_fcf = df_fcf["FCF_B"].notna()
        if valid_fcf.any():
            fcf_labels = df_fcf.loc[valid_fcf, "Period"].dt.strftime("%Y-%m-%d")
            ax3.bar(fcf_labels, df_fcf.loc[valid_fcf, "FCF_B"], width=0.55, alpha=0.85, label="Free Cash Flow ($B)")
            ax3.set_title("Standalone Quarterly Free Cash Flow ($B) & Growth", fontweight="bold", fontsize=10.5)
            ax3.set_ylabel("FCF ($B)")
            ax3.tick_params(axis="x", rotation=45, labelsize=7)

            ax3_sub = ax3.twinx()
            ax3_sub.plot(fcf_labels, df_fcf.loc[valid_fcf, "FCF_YoY_%"], marker="o", linewidth=1.5, label="YoY Growth (%)")
            ax3_sub.plot(fcf_labels, df_fcf.loc[valid_fcf, "FCF_QoQ_%"], marker="s", linestyle="--", linewidth=1.2, label="QoQ Growth (%)")
            ax3_sub.set_ylabel("Growth (%)")
            ax3_sub.grid(False)

            h1, l1 = ax3.get_legend_handles_labels()
            h2, l2 = ax3_sub.get_legend_handles_labels()
            ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.5)

        # Margins
        ax4.plot(xlabels, df_raw["Op_Margin_%"], marker="o", linewidth=2, label="Operating Margin (%)")
        ax4.plot(xlabels, df_raw["Net_Margin_%"], marker="s", linestyle="--", linewidth=2, label="Net Margin (%)")
        ax4.axhline(0, linestyle=":", linewidth=1, alpha=0.6)
        ax4.set_title("Operating Margin vs. Net Margin (%)", fontweight="bold", fontsize=10.5)
        ax4.set_ylabel("Margin (%)")
        ax4.tick_params(axis="x", rotation=45, labelsize=7)
        ax4.legend(loc="upper left", fontsize=7)
        ax4.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # CHART 2: VALUATION / PRICE
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Valuation & Price Action")

        fig2, ((ax_p1, ax_p2), (ax_p3, ax_p4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
        fig2.suptitle(f"{ticker_symbol} Price Action & Valuation", fontsize=15, fontweight="bold", y=0.98)

        # Price
        if hist_price is not None and not hist_price.empty:
            ax_p1.plot(hist_price.index, hist_price["Close"], linewidth=1.5, label="Close Price ($)")
            ax_p1.plot(hist_price.index, hist_price["EMA50"], linewidth=1.2, label="50-Day EMA")
            ax_p1.plot(hist_price.index, hist_price["EMA200"], linestyle="--", linewidth=1.2, label="200-Day EMA")

        ax_p1.set_title("Daily Stock Price vs 50/200 EMA", fontweight="bold", fontsize=10.5)
        ax_p1.set_ylabel("Price ($)")
        ax_p1.legend(loc="upper left", fontsize=7)
        ax_p1.grid(True, linestyle="--", alpha=0.3)

        # P/S
        if "P_S_TTM" in df_raw.columns:
            ax_p2.plot(xlabels, df_raw["P_S_TTM"], marker="o", linewidth=2, label="P/S (TTM)")

        ax_p2.set_title("Price-to-Sales (P/S) — TTM", fontweight="bold", fontsize=10.5)
        ax_p2.set_ylabel("P/S Multiple (x)")
        ax_p2.tick_params(axis="x", rotation=45, labelsize=7)
        ax_p2.legend(loc="upper left", fontsize=7)
        ax_p2.grid(True, linestyle="--", alpha=0.3)

        # P/E
        if "P_E_TTM" in df_raw.columns:
            ax_p2_pe = ax_p3
            ax_p2_pe.plot(xlabels, df_raw["P_E_TTM"], marker="s", linewidth=2, label="P/E (TTM)")
            ax_p2_pe.axhline(0, linestyle=":", linewidth=1, alpha=0.6)
            ax_p2_pe.set_title("Price-to-Earnings (P/E) — TTM", fontweight="bold", fontsize=10.5)
            ax_p2_pe.set_ylabel("P/E Multiple (x)")
            ax_p2_pe.tick_params(axis="x", rotation=45, labelsize=7)
            ax_p2_pe.legend(loc="upper left", fontsize=7)
            ax_p2_pe.grid(True, linestyle="--", alpha=0.3)

        # FCF yield
        if "FCF_Yield_%" in df_raw.columns:
            ax_p4.plot(xlabels, df_raw["FCF_Yield_%"], marker="^", linewidth=2, label="FCF Yield (TTM %)")
            ax_p4.axhline(0, linestyle=":", linewidth=1, alpha=0.6)
            ax_p4.set_title("Free Cash Flow Yield — TTM", fontweight="bold", fontsize=10.5)
            ax_p4.set_ylabel("FCF Yield (%)")
            ax_p4.tick_params(axis="x", rotation=45, labelsize=7)
            ax_p4.legend(loc="upper left", fontsize=7)
            ax_p4.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ----------------------------------------------------
        # CHART 3: CAPEX / SHARES
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader(f"{ticker_symbol} — Capex & Share Count")

        fig3, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)

        ax_c1.bar(xlabels, df_raw["Capex"].abs() / 1e9, width=0.55, alpha=0.85, label="Capex ($B)")
        ax_c1.set_title("Quarterly Capital Expenditures ($B)", fontweight="bold", fontsize=10.5)
        ax_c1.set_ylabel("Capex ($B)")
        ax_c1.tick_params(axis="x", rotation=45, labelsize=7)
        ax_c1.legend(loc="upper left", fontsize=7)
        ax_c1.grid(True, linestyle="--", alpha=0.3)

        ax_c2.plot(xlabels, df_raw["Share_Dilution_YoY_%"], marker="o", linewidth=2, label="Diluted Share Growth YoY (%)")
        ax_c2.axhline(0, linestyle=":", linewidth=1, alpha=0.6)
        ax_c2.set_title("Diluted Share Count Change — YoY %", fontweight="bold", fontsize=10.5)
        ax_c2.set_ylabel("Share Count YoY Change (%)")
        ax_c2.tick_params(axis="x", rotation=45, labelsize=7)
        ax_c2.legend(loc="upper left", fontsize=7)
        ax_c2.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as exc:
        st.error(f"Could not load data for ticker '{ticker_symbol}'. Error: {type(exc).__name__}: {exc}")
        st.exception(exc)
