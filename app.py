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

    run_button = st.button("Generate Report", type="primary")

    st.markdown("---")
    st.caption(
        "SEC XBRL is the primary financial-statement source. "
        "The parser uses individual XBRL facts and their actual reporting "
        "periods instead of taking the first year-looking table column."
    )
    st.caption(
        "Important: review the raw dataset for unusual reporting items "
        "before relying on any derived valuation metric."
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
        "RevenueFromContractsWithCustomers",  # IFRS
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Revenue",  # IFRS generic
    ],
    "Operating_Income": [
        "OperatingIncomeLoss",
        "ProfitLossFromOperatingActivities",  # IFRS
    ],
    "Net_Income": [
        "NetIncomeLoss",
        "ProfitLoss",  # IFRS
        "ProfitLossAttributableToOwners",  # IFRS
    ],
    "Diluted_EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "DilutedEarningsPerShare",  # IFRS
    ],
    "Diluted_Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ],
    "OCF": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashFlowsFromUsedInOperatingActivities",  # IFRS
    ],
    "Capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PurchaseOfPropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndOtherPropertyPlantAndEquipment",
        "AdditionsToPropertyPlantAndEquipment",  # IFRS
    ],
}

LABEL_FALLBACKS = {
    "Revenue": [
        "revenue",
        "sales",
    ],
    "Operating_Income": [
        "operating profit",
        "operating income",
    ],
    "Net_Income": [
        "profit loss",
        "profit for the period",
        "net income",
    ],
    "Diluted_EPS": [
        "diluted earnings per share",
    ],
    "Diluted_Shares": [
        "weighted average number of diluted shares",
    ],
    "OCF": [
        "cash flows from operating activities",
        "net cash from operating activities",
    ],
    "Capex": [
        "purchase of property plant and equipment",
        "payments to acquire property plant and equipment",
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

        q_end = quarter_from_frame(frame)

        if q_end is not None:
            output.append(
                {
                    "entry": e,
                    "period": q_end,
                    "method": "SEC frame",
                    "score": 100,
                }
            )
            continue

        if 70 <= days <= 110:
            end = entry_date(e, "end")
            score = 50

            if form in {"10-Q", "20-F", "6-K"}:
                score += 10

            output.append(
                {
                    "entry": e,
                    "period": end,
                    "method": "duration",
                    "score": score,
                }
            )

    return output


def annual_candidates(entries):
    output = []

    for e in entries:
        if not entry_is_flow(e):
            continue

        days = entry_duration_days(e)
        if days is None:
            continue

        form = str(e.get("form", "")).upper()

        if 320 <= days <= 390 and form in {"10-K", "20-F"}:
            output.append(e)

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

            quarter_rows[key][field] = clean_number(
                entry.get("val")
            )

            quarter_rows[key][f"{field}_Concept"] = (
                f"{meta['taxonomy']}:{meta['concept']}"
            )

            quarter_rows[key][f"{field}_Method"] = best["method"]

    df = pd.DataFrame(list(quarter_rows.values()))

    if df.empty:
        return df, fact_meta

    df["Period"] = pd.to_datetime(df["Period"], errors="coerce")

    df = (
        df.dropna(subset=["Period"])
        .sort_values("Period")
        .drop_duplicates("Period", keep="last")
        .reset_index(drop=True)
    )

    df = df[df["Period"] >= pd.Timestamp("2017-01-01")].reset_index(drop=True)

    return df, fact_meta


# ============================================================
# YAHOO FALLBACK / SUPPLEMENT
# ============================================================

YF_KEYS = {
    "Revenue": [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
    ],
    "Operating_Income": [
        "Operating Income",
        "Operating Income Loss",
        "EBIT",
    ],
    "Net_Income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ],
    "Diluted_EPS": [
        "Diluted EPS",
        "Diluted EPS From Continuing Operations",
    ],
    "Diluted_Shares": [
        "Diluted Average Shares",
        "Diluted Average Shares Outstanding",
    ],
    "OCF": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "Capex": [
        "Capital Expenditure",
        "Capital Expenditure Reported",
        "Purchase Of Property Plant And Equipment",
    ],
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
                    "Revenue": yahoo_row_value(
                        inc, YF_KEYS["Revenue"], date_col
                    ),
                    "Operating_Income": yahoo_row_value(
                        inc, YF_KEYS["Operating_Income"], date_col
                    ),
                    "Net_Income": yahoo_row_value(
                        inc, YF_KEYS["Net_Income"], date_col
                    ),
                    "Diluted_EPS": yahoo_row_value(
                        inc, YF_KEYS["Diluted_EPS"], date_col
                    ),
                    "Diluted_Shares": yahoo_row_value(
                        inc, YF_KEYS["Diluted_Shares"], date_col
                    ),
                    "OCF": yahoo_row_value(
                        cf, YF_KEYS["OCF"], date_col
                    ),
                    "Capex": yahoo_row_value(
                        cf, YF_KEYS["Capex"], date_col
                    ),
                }
            )

        return pd.DataFrame(rows), diagnostics

    except Exception as exc:
        return pd.DataFrame(), [
            f"Yahoo error: {type(exc).__name__}: {exc}"
        ]


def merge_sec_yahoo(sec_df, yahoo_df):
    if sec_df.empty:
        return yahoo_df.copy()

    if yahoo_df.empty:
        return sec_df.copy()

    cols = [
        "Period",
        "Source",
        "Revenue",
        "Operating_Income",
        "Net_Income",
        "Diluted_EPS",
        "Diluted_Shares",
        "OCF",
        "Capex",
    ]

    for df in [sec_df, yahoo_df]:
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan

    sec = sec_df.copy()
    yf_df = yahoo_df.copy()

    sec["Period"] = pd.to_datetime(sec["Period"], errors="coerce")
    yf_df["Period"] = pd.to_datetime(yf_df["Period"], errors="coerce")

    sec_indexed = sec.set_index("Period")
    yf_indexed = yf_df.set_index("Period")

    all_periods = sorted(
        set(sec_indexed.index.dropna())
        | set(yf_indexed.index.dropna())
    )

    output = []

    for period in all_periods:
        if period in sec_indexed.index:
            row = sec_indexed.loc[period]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            result = {
                "Period": period,
                "Source": "SEC XBRL",
            }

            for col in cols[2:]:
                value = clean_number(row.get(col, np.nan))

                if pd.notna(value):
                    result[col] = value
                elif period in yf_indexed.index:
                    yfrow = yf_indexed.loc[period]

                    if isinstance(yfrow, pd.DataFrame):
                        yfrow = yfrow.iloc[-1]

                    result[col] = clean_number(
                        yfrow.get(col, np.nan)
                    )
                    if pd.notna(result[col]):
                        result["Source"] = "SEC + Yahoo"
                else:
                    result[col] = np.nan

            for c in sec.columns:
                if c.endswith("_Concept") or c.endswith("_Method"):
                    result[c] = row.get(c, None)

            output.append(result)

        else:
            yfrow = yf_indexed.loc[period]

            if isinstance(yfrow, pd.DataFrame):
                yfrow = yfrow.iloc[-1]

            result = {
                "Period": period,
                "Source": "Yahoo Finance",
            }

            for col in cols[2:]:
                result[col] = clean_number(
                    yfrow.get(col, np.nan)
                )

            output.append(result)

    return (
        pd.DataFrame(output)
        .sort_values("Period")
        .reset_index(drop=True)
    )


# ============================================================
# FINANCIAL DERIVATIONS
# ============================================================

def calculate_metrics(df):
    work = df.copy()

    numeric_cols = [
        "Revenue",
        "Operating_Income",
        "Net_Income",
        "Diluted_EPS",
        "Diluted_Shares",
        "OCF",
        "Capex",
    ]

    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["Revenue_B"] = work["Revenue"] / 1e9

    work["Rev_YoY_%"] = pct_change_safe(
        work["Revenue_B"], 4
    )

    work["Rev_QoQ_%"] = pct_change_safe(
        work["Revenue_B"], 1
    )

    work["EPS_YoY_%"] = pct_change_safe(
        work["Diluted_EPS"], 4
    ).clip(-500, 500)

    work["EPS_QoQ_%"] = pct_change_safe(
        work["Diluted_EPS"], 1
    ).clip(-500, 500)

    work["Op_Margin_%"] = (
        safe_divide(
            work["Operating_Income"],
            work["Revenue"],
        )
        * 100
    )

    work["Net_Margin_%"] = (
        safe_divide(
            work["Net_Income"],
            work["Revenue"],
        )
        * 100
    )

    work["Diluted_Shares_M"] = (
        work["Diluted_Shares"] / 1e6
    )

    work["Share_Dilution_YoY_%"] = pct_change_safe(
        work["Diluted_Shares_M"], 4
    ).clip(-50, 50)

    return work


def calculate_fcf(df):
    work = df[
        ["Period", "OCF", "Capex", "Source"]
    ].copy()

    work["OCF"] = pd.to_numeric(
        work["OCF"], errors="coerce"
    )

    work["Capex"] = pd.to_numeric(
        work["Capex"], errors="coerce"
    )

    work["Capex_Positive"] = work["Capex"].abs()

    work["FCF"] = (
        work["OCF"] - work["Capex_Positive"]
    )

    work["FCF_B"] = work["FCF"] / 1e9

    work["FCF_YoY_%"] = pct_change_safe(
        work["FCF"], 4
    ).clip(-500, 500)

    work["FCF_QoQ_%"] = pct_change_safe(
        work["FCF"], 1
    ).clip(-500, 500)

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

        shares_outstanding = info.get(
            "sharesOutstanding"
        )

        market_cap = info.get("marketCap")

        hist = tk.history(
            period="2y",
            auto_adjust=False,
        )

        if (
            current_price is None
            and hist is not None
            and not hist.empty
        ):
            current_price = clean_number(
                hist["Close"].iloc[-1]
            )

        if (
            market_cap is None
            and current_price is not None
            and shares_outstanding is not None
        ):
            market_cap = (
                current_price * shares_outstanding
            )

        if hist is not None and not hist.empty:
            hist = hist.copy()

            hist["EMA50"] = (
                hist["Close"]
                .ewm(span=50, adjust=False)
                .mean()
            )

            hist["EMA200"] = (
                hist["Close"]
                .ewm(span=200, adjust=False)
                .mean()
            )

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
def fetch_and_parse_ticker(ticker):
    diagnostics = []

    cik = get_cik_for_ticker(ticker)
    companyfacts = get_companyfacts(cik)

    sec_df, fact_meta = build_quarterly_fact_table(
        companyfacts
    )

    if sec_df.empty:
        diagnostics.append(
            "SEC Company Facts did not yield standalone quarterly flow facts."
        )

    yahoo_df, yahoo_diag = fetch_yahoo_quarterly(ticker)

    diagnostics.extend(yahoo_diag)

    df = merge_sec_yahoo(
        sec_df,
        yahoo_df,
    )

    if df.empty:
        raise ValueError(
            f"No usable quarterly data found for {ticker}."
        )

    df = calculate_metrics(df)

    implied = safe_divide(
        df["Net_Income"],
        df["Diluted_EPS"],
    )

    missing = (
        df["Diluted_Shares"].isna()
        & implied.notna()
        & (df["Diluted_EPS"].abs() > 0.0001)
    )

    df.loc[missing, "Diluted_Shares"] = implied[missing]
    df.loc[missing, "Diluted_Shares_Method"] = (
        "Implied Net Income / Diluted EPS"
    )

    df["Diluted_Shares_M"] = (
        df["Diluted_Shares"] / 1e6
    )

    df["Share_Dilution_YoY_%"] = pct_change_safe(
        df["Diluted_Shares_M"], 4
    ).clip(-50, 50)

    return df, fact_meta, diagnostics


# ============================================================
# RUN
# ============================================================

if run_button or ticker_symbol:
    try:
        with st.spinner(
            f"Loading SEC XBRL and market data for {ticker_symbol}..."
        ):
            (
                df_all,
                fact_meta,
                diagnostics,
            ) = fetch_and_parse_ticker(
                ticker_symbol
            )

            (
                hist_price,
                current_price,
                market_cap,
                shares_outstanding,
            ) = fetch_market_data(
                ticker_symbol
            )

        df_raw = (
            df_all
            .tail(lookback_quarters)
            .reset_index(drop=True)
        )

        df_fcf = calculate_fcf(df_raw)

        # ----------------------------------------------------
        # TTM VALUATION
        # ----------------------------------------------------

        df_raw["TTM_Revenue"] = (
            df_raw["Revenue"].rolling(4).sum()
        )

        df_raw["TTM_EPS"] = (
            df_raw["Diluted_EPS"].rolling(4).sum()
        )

        df_fcf["TTM_FCF"] = (
            df_fcf["FCF"].rolling(4).sum()
        )

        if (
            pd.notna(market_cap)
            and market_cap > 0
            and pd.notna(current_price)
            and current_price > 0
        ):
            df_raw["P_S_TTM"] = safe_divide(
                market_cap,
                df_raw["TTM_Revenue"],
            ).clip(lower=0, upper=150)

            df_raw["P_E_TTM"] = safe_divide(
                current_price,
                df_raw["TTM_EPS"],
            ).where(
                df_raw["TTM_EPS"] > 0
            ).clip(lower=0, upper=250)

            df_raw["TTM_FCF"] = (
                df_fcf["TTM_FCF"].values
            )

            df_raw["FCF_Yield_%"] = (
                df_raw["TTM_FCF"]
                / market_cap
                * 100
            )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        st.subheader(
            f"{ticker_symbol} — Executive Financial Dashboard"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Current Price",
                (
                    f"${current_price:,.2f}"
                    if pd.notna(current_price)
                    else "N/A"
                ),
            )

        with c2:
            st.metric(
                "Market Cap",
                (
                    f"${market_cap / 1e9:,.2f}B"
                    if pd.notna(market_cap)
                    else "N/A"
                ),
            )

        with c3:
            st.metric(
                "SEC / XBRL Quarters",
                str(
                    df_all["Source"]
                    .astype(str)
                    .str.contains("SEC")
                    .sum()
                ),
            )

        with c4:
            st.metric(
                "Quarterly Records",
                str(len(df_all)),
            )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        with st.expander(
            "Data Source Diagnostics",
            expanded=False,
        ):
            st.write("**Quarterly source by period:**")

            diag_cols = [
                "Period",
                "Source",
            ]

            for col in [
                "Revenue_Concept",
                "Revenue_Method",
                "Diluted_EPS_Concept",
                "Diluted_EPS_Method",
                "Diluted_Shares_Concept",
                "Diluted_Shares_Method",
                "OCF_Concept",
                "OCF_Method",
                "Capex_Concept",
                "Capex_Method",
            ]:
                if col in df_all.columns:
                    diag_cols.append(col)

            st.dataframe(
                df_all[diag_cols]
                .tail(20),
                use_container_width=True,
            )

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
                    st.dataframe(
                        pd.DataFrame(concept_rows),
                        use_container_width=True,
                    )

            if diagnostics:
                st.write("**Diagnostics:**")
                for message in diagnostics[-20:]:
                    st.caption(message)

        # ----------------------------------------------------
        # RAW TABLE
        # ----------------------------------------------------

        with st.expander(
            "Quarterly Financial Dataset",
            expanded=False,
        ):
            display = df_raw.copy()
            display["Period"] = display[
                "Period"
            ].dt.strftime("%Y-%m-%d")

            display_cols = [
                "Period",
                "Source",
                "Revenue_B",
                "Rev_YoY_%",
                "Diluted_EPS",
                "EPS_YoY_%",
                "Operating_Income",
                "Op_Margin_%",
                "Net_Income",
                "Net_Margin_%",
                "OCF",
                "Capex",
                "Diluted_Shares_M",
                "Share_Dilution_YoY_%",
            ]

            display_cols = [
                c for c in display_cols
                if c in display.columns
            ]

            st.dataframe(
                display[display_cols],
                use_container_width=True,
            )

        # ----------------------------------------------------
        # CHART 1: FINANCIALS
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Financial Performance"
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16, 11),
            dpi=150,
        )

        fig.suptitle(
            f"{ticker_symbol} Financial & Growth Dashboard",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        xlabels = df_raw["Period"].dt.strftime(
            "%Y-%m-%d"
        )

        # Revenue
        ax = axes[0, 0]
        valid = df_raw["Revenue_B"].notna()

        if valid.any():
            ax.bar(
                xlabels[valid],
                df_raw.loc[valid, "Revenue_B"],
                width=0.55,
                alpha=0.85,
                label="Revenue ($B)",
            )

            ax.set_title(
                "Revenue ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax.set_ylabel("Revenue ($B)")
            ax.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax2 = ax.twinx()

            ax2.plot(
                xlabels[valid],
                df_raw.loc[valid, "Rev_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax2.plot(
                xlabels[valid],
                df_raw.loc[valid, "Rev_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax2.set_ylabel("Growth (%)")
            ax2.grid(False)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            ax.legend(
                h1 + h2,
                l1 + l2,
                loc="upper left",
                fontsize=6.5,
            )

        # EPS
        ax = axes
        valid = df_raw["Diluted_EPS"].notna()

        if valid.any():
            ax.bar(
                xlabels[valid],
                df_raw.loc[valid, "Diluted_EPS"],
                width=0.55,
                alpha=0.85,
                label="Diluted EPS ($)",
            )

            ax.set_title(
                "Diluted EPS ($) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax.set_ylabel("EPS ($)")
            ax.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax2 = ax.twinx()

            ax2.plot(
                xlabels[valid],
                df_raw.loc[valid, "EPS_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax2.plot(
                xlabels[valid],
                df_raw.loc[valid, "EPS_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax2.set_ylabel("Growth (%)")
            ax2.grid(False)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            ax.legend(
                h1 + h2,
                l1 + l2,
                loc="upper left",
                fontsize=6.5,
            )

        # FCF
        ax = axes
        valid = df_fcf["FCF_B"].notna()

        if valid.any():
            fcf_labels = df_fcf.loc[
                valid, "Period"
            ].dt.strftime("%Y-%m-%d")

            ax.bar(
                fcf_labels,
                df_fcf.loc[valid, "FCF_B"],
                width=0.55,
                alpha=0.85,
                label="Free Cash Flow ($B)",
            )

            ax.set_title(
                "Standalone Quarterly Free Cash Flow ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax.set_ylabel("FCF ($B)")
            ax.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax2 = ax.twinx()

            ax2.plot(
                fcf_labels,
                df_fcf.loc[valid, "FCF_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax2.plot(
                fcf_labels,
                df_fcf.loc[valid, "FCF_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax2.set_ylabel("Growth (%)")
            ax2.grid(False)

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            ax.legend(
                h1 + h2,
                l1 + l2,
                loc="upper left",
                fontsize=6.5,
            )

        # Margins
        ax = axes

        ax.plot(
            xlabels,
            df_raw["Op_Margin_%"],
            marker="o",
            linewidth=2,
            label="Operating Margin (%)",
        )

        ax.plot(
            xlabels,
            df_raw["Net_Margin_%"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Net Margin (%)",
        )

        ax.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax.set_title(
            "Operating Margin vs. Net Margin (%)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("Margin (%)")
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # CHART 2: VALUATION / PRICE
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Valuation & Price Action"
        )

        fig2, axes2 = plt.subplots(
            2,
            2,
            figsize=(16, 11),
            dpi=150,
        )

        fig2.suptitle(
            f"{ticker_symbol} Price Action & Valuation",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Price
        ax = axes2[0, 0]

        if (
            hist_price is not None
            and not hist_price.empty
        ):
            ax.plot(
                hist_price.index,
                hist_price["Close"],
                linewidth=1.5,
                label="Close Price ($)",
            )

            ax.plot(
                hist_price.index,
                hist_price["EMA50"],
                linewidth=1.2,
                label="50-Day EMA",
            )

            ax.plot(
                hist_price.index,
                hist_price["EMA200"],
                linestyle="--",
                linewidth=1.2,
                label="200-Day EMA",
            )

        ax.set_title(
            "Daily Stock Price vs 50/200 EMA",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("Price ($)")
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/S
        ax = axes2

        if "P_S_TTM" in df_raw.columns:
            ax.plot(
                xlabels,
                df_raw["P_S_TTM"],
                marker="o",
                linewidth=2,
                label="P/S (TTM)",
            )

        ax.set_title(
            "Price-to-Sales (P/S) — TTM",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("P/S Multiple (x)")
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/E
        ax = axes2

        if "P_E_TTM" in df_raw.columns:
            ax.plot(
                xlabels,
                df_raw["P_E_TTM"],
                marker="s",
                linewidth=2,
                label="P/E (TTM)",
            )

        ax.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax.set_title(
            "Price-to-Earnings (P/E) — TTM",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("P/E Multiple (x)")
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # FCF yield
        ax = axes2

        if "FCF_Yield_%" in df_raw.columns:
            ax.plot(
                xlabels,
                df_raw["FCF_Yield_%"],
                marker="^",
                linewidth=2,
                label="FCF Yield (TTM %)",
            )

        ax.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax.set_title(
            "Free Cash Flow Yield — TTM",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("FCF Yield (%)")
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout()

        st.pyplot(fig2)
        plt.close(fig2)

        # ----------------------------------------------------
        # CHART 3: CAPEX / SHARES
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Capex & Share Count"
        )

        fig3, axes3 = plt.subplots(
            1,
            2,
            figsize=(16, 5),
            dpi=150,
        )

        ax = axes3[0]

        ax.bar(
            xlabels,
            df_raw["Capex"].abs() / 1e9,
            width=0.55,
            alpha=0.85,
            label="Capex ($B)",
        )

        ax.set_title(
            "Quarterly Capital Expenditures ($B)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("Capex ($B)")
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        ax = axes3

        ax.plot(
            xlabels,
            df_raw["Share_Dilution_YoY_%"],
            marker="o",
            linewidth=2,
            label="Diluted Share Growth YoY (%)",
        )

        ax.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax.set_title(
            "Diluted Share Count Change — YoY %",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_ylabel("Share Count YoY Change (%)")
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax.legend(
            loc="upper left",
            fontsize=7,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout()

        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as exc:
        st.error(
            f"Could not load data for ticker '{ticker_symbol}'. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        st.exception(exc)
