from pathlib import Path
import os
import re
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from edgar import Company, set_identity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

EDGAR_IDENTITY = os.getenv(
    "EDGAR_IDENTITY",
    st.secrets.get("EDGAR_IDENTITY", "Scott Davis scott@example.com"),
)
set_identity(EDGAR_IDENTITY)

st.set_page_config(
    page_title="SEC XBRL Financial Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Institutional SEC XBRL Financial Terminal")
st.markdown(
    "Enter a stock ticker to pull standardized quarterly financial data, "
    "standalone quarterly cash flow, valuation history, and executive charts."
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

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
        "**Data hierarchy:** SEC XBRL/EdgarTools first; Yahoo Finance is used "
        "as a fallback for financial data and for market-price history."
    )
    st.caption(
        "**Audit advisory:** Cross-reference important XBRL values against "
        "the company's official SEC filing before making investment decisions."
    )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def clean_number(value):
    """Convert common SEC/Yahoo numeric representations to float."""
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if pd.notna(value) else np.nan

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


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def is_reasonable_numeric(x):
    return pd.notna(x) and np.isfinite(x)


def pct_change_safe(series, periods):
    return series.pct_change(periods=periods, fill_method=None) * 100


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


# ---------------------------------------------------------------------------
# XBRL / SEC EXTRACTION
# ---------------------------------------------------------------------------

CONCEPT_CANDIDATES = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Revenue",
    ],
    "Operating_Income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "Net_Income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "Diluted_EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "BasicAndDilutedEarningsPerShare",
    ],
    "Diluted_Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ],
    "OCF": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "Capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForProceedsFromOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}


def find_statement_value(statement_df, candidates):
    """Find a statement value using standardized XBRL concept/label information."""
    if statement_df is None or statement_df.empty:
        return np.nan

    df = statement_df.copy()

    concept_col = None
    for c in ["standard_concept", "concept", "label", "name"]:
        if c in df.columns:
            concept_col = c
            break

    if concept_col is None:
        working = df.copy()
        working["_row_text"] = working.index.astype(str)
    else:
        working = df.copy()
        working["_row_text"] = working[concept_col].astype(str)

    for candidate in candidates:
        candidate_norm = normalize_label(candidate)

        exact = working[
            working["_row_text"].map(normalize_label) == candidate_norm
        ]
        if not exact.empty:
            row = exact.iloc[0]
            value = _latest_numeric_period_value(row)
            if is_reasonable_numeric(value):
                return value

    candidate_words = [normalize_label(x) for x in candidates]
    mask = working["_row_text"].map(
        lambda x: any(cw in normalize_label(x) for cw in candidate_words)
    )
    matches = working[mask]

    if len(matches) == 1:
        return _latest_numeric_period_value(matches.iloc[0])

    return np.nan


def _latest_numeric_period_value(row):
    """Select a numeric value from a statement row."""
    candidates = []

    for col in row.index:
        if str(col).startswith("_"):
            continue

        value = clean_number(row[col])
        if not is_reasonable_numeric(value):
            continue

        text = str(col)
        if any(
            x in text.lower()
            for x in ["label", "concept", "standard_concept", "units"]
        ):
            continue

        score = 0
        upper = text.upper()

        if "Q" in upper:
            score += 10
        if "FY" in upper:
            score += 3
        if "202" in upper:
            score += 2

        candidates.append((score, text, value))

    if not candidates:
        return np.nan

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _period_from_filing(filing):
    """Get filing report date safely."""
    try:
        return pd.to_datetime(str(filing.period_of_report)[:10], errors="coerce")
    except Exception:
        return pd.NaT


def _extract_from_edgar_statements(ticker):
    """Primary SEC path."""
    records = []
    diagnostics = []

    try:
        company = Company(ticker)
        filings = company.get_filings(form=["10-Q", "10-K", "20-F"])
        filings = list(filings)[:60]

        for filing in filings:
            period = _period_from_filing(filing)
            if pd.isna(period):
                continue

            form = str(getattr(filing, "form", ""))

            try:
                obj = filing.obj()

                inc = getattr(obj, "income_statement", None)
                cf = getattr(obj, "cash_flow_statement", None)

                inc_df = (
                    inc.to_dataframe(view="standard")
                    if inc is not None
                    else None
                )
                cf_df = (
                    cf.to_dataframe(view="standard") if cf is not None else None
                )

                if inc_df is None or inc_df.empty:
                    continue

                record = {
                    "Period": period,
                    "Form": form,
                    "Source": "SEC XBRL",
                    "Revenue": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Revenue"]
                    ),
                    "Operating_Income": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Operating_Income"]
                    ),
                    "Net_Income": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Net_Income"]
                    ),
                    "Diluted_EPS": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Diluted_EPS"]
                    ),
                    "Diluted_Shares": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Diluted_Shares"]
                    ),
                    "OCF": find_statement_value(
                        cf_df, CONCEPT_CANDIDATES["OCF"]
                    ),
                    "Capex": find_statement_value(
                        cf_df, CONCEPT_CANDIDATES["Capex"]
                    ),
                }

                if is_reasonable_numeric(record["Revenue"]) or is_reasonable_numeric(
                    record["Net_Income"]
                ):
                    records.append(record)

            except Exception as exc:
                diagnostics.append(
                    f"{period.date()} {form}: {type(exc).__name__}: {exc}"
                )
                continue

    except Exception as exc:
        diagnostics.append(
            f"SEC company/fillings error: {type(exc).__name__}: {exc}"
        )

    if not records:
        return pd.DataFrame(), diagnostics

    df = pd.DataFrame(records)
    df["completeness"] = df.notna().sum(axis=1)

    df = (
        df.sort_values(["Period", "completeness"], ascending=[True, False])
        .drop_duplicates("Period", keep="first")
        .drop(columns=["completeness"])
        .reset_index(drop=True)
    )

    return df, diagnostics


# ---------------------------------------------------------------------------
# YAHOO FALLBACK / SUPPLEMENT
# ---------------------------------------------------------------------------

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
        "Basic EPS",
    ],
    "Diluted_Shares": [
        "Diluted Average Shares",
        "Diluted Average Shares Outstanding",
        "Basic Average Shares",
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
                value = df.loc[key, date_col]
                return clean_number(value)
            except Exception:
                continue
    return np.nan


def fetch_yahoo_quarterly(ticker):
    """Yahoo fallback."""
    diagnostics = []

    try:
        tk = yf.Ticker(ticker)

        inc = tk.quarterly_income_stmt
        cf = tk.quarterly_cashflow

        if inc is None or inc.empty:
            return pd.DataFrame(), ["Yahoo quarterly income statement empty"]

        records = []

        for date_col in inc.columns:
            period = pd.to_datetime(date_col, errors="coerce")
            if pd.isna(period):
                continue

            records.append(
                {
                    "Period": period,
                    "Form": "Yahoo",
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
                    "OCF": yahoo_row_value(cf, YF_KEYS["OCF"], date_col),
                    "Capex": yahoo_row_value(cf, YF_KEYS["Capex"], date_col),
                }
            )

        return pd.DataFrame(records), diagnostics

    except Exception as exc:
        diagnostics.append(f"Yahoo error: {type(exc).__name__}: {exc}")
        return pd.DataFrame(), diagnostics


def merge_sec_and_yahoo(sec_df, yf_df):
    """SEC is authoritative where a value exists. Yahoo fills missing fields."""
    if sec_df.empty:
        return yf_df.copy()

    if yf_df.empty:
        return sec_df.copy()

    all_cols = [
        "Period",
        "Form",
        "Source",
        "Revenue",
        "Operating_Income",
        "Net_Income",
        "Diluted_EPS",
        "Diluted_Shares",
        "OCF",
        "Capex",
    ]

    sec = sec_df.copy()
    yahoo = yf_df.copy()

    for df in [sec, yahoo]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan

    combined = pd.concat(
        [sec[all_cols], yahoo[all_cols]],
        ignore_index=True,
    )

    combined["source_priority"] = np.where(
        combined["Source"].eq("SEC XBRL"), 0, 1
    )

    combined = combined.sort_values(["Period", "source_priority"])

    output = []

    for period, group in combined.groupby("Period", sort=True):
        row = {
            "Period": period,
            "Form": group.iloc[0]["Form"],
            "Source": group.iloc[0]["Source"],
        }

        for col in [
            "Revenue",
            "Operating_Income",
            "Net_Income",
            "Diluted_EPS",
            "Diluted_Shares",
            "OCF",
            "Capex",
        ]:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[col] = values.iloc[0] if not values.empty else np.nan

        if len(group) > 1:
            sec_row = group[group["Source"].eq("SEC XBRL")]
            yf_row = group[group["Source"].eq("Yahoo Finance")]

            if not sec_row.empty and not yf_row.empty:
                for col in [
                    "Revenue",
                    "Operating_Income",
                    "Net_Income",
                    "Diluted_EPS",
                    "Diluted_Shares",
                    "OCF",
                    "Capex",
                ]:
                    sec_value = pd.to_numeric(
                        sec_row.iloc[0][col], errors="coerce"
                    )
                    yf_value = pd.to_numeric(
                        yf_row.iloc[0][col], errors="coerce"
                    )

                    if pd.isna(sec_value) and pd.notna(yf_value):
                        row[col] = yf_value
                        row["Source"] = "SEC + Yahoo"

        output.append(row)

    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# CASH FLOW NORMALIZATION
# ---------------------------------------------------------------------------


def calculate_fcf_safe(df):
    """Calculate standalone FCF."""
    if df.empty:
        return pd.DataFrame(columns=["Period", "FCF", "FCF_B"])

    work = df[["Period", "OCF", "Capex"]].copy()
    work["Period"] = pd.to_datetime(work["Period"], errors="coerce")
    work["OCF"] = pd.to_numeric(work["OCF"], errors="coerce")
    work["Capex"] = pd.to_numeric(work["Capex"], errors="coerce")
    work = work.sort_values("Period").reset_index(drop=True)

    standalone_ocf = []
    standalone_capex = []

    prev_year = None
    prev_ocf_ytd = np.nan
    prev_capex_ytd = np.nan

    for _, row in work.iterrows():
        p_date = row["Period"]
        curr_year = p_date.year if pd.notna(p_date) else None
        month = p_date.month if pd.notna(p_date) else 3

        ocf = row["OCF"]
        capex = row["Capex"]

        if curr_year != prev_year or month <= 4:
            q_ocf = ocf
            q_capex = capex
        else:
            q_ocf = ocf - prev_ocf_ytd if pd.notna(prev_ocf_ytd) else ocf
            q_capex = capex - prev_capex_ytd if pd.notna(prev_capex_ytd) else capex

        standalone_ocf.append(q_ocf)
        standalone_capex.append(q_capex)

        prev_year = curr_year
        prev_ocf_ytd = ocf
        prev_capex_ytd = capex

    fcf = np.array(standalone_ocf, dtype=float) - np.abs(
        np.array(standalone_capex, dtype=float)
    )

    out = pd.DataFrame(
        {
            "Period": work["Period"],
            "FCF": fcf,
        }
    )

    out["FCF_B"] = out["FCF"] / 1e9
    out["FCF_YoY_%"] = pct_change_safe(out["FCF"], 4).clip(-200, 200)
    out["FCF_QoQ_%"] = pct_change_safe(out["FCF"], 1).clip(-200, 200)

    return out


# ---------------------------------------------------------------------------
# MAIN DATA FETCH
# ---------------------------------------------------------------------------


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_and_parse_ticker(ticker):
    sec_df, sec_diag = _extract_from_edgar_statements(ticker)
    yf_df, yf_diag = fetch_yahoo_quarterly(ticker)

    df = merge_sec_and_yahoo(sec_df, yf_df)

    if df.empty:
        raise ValueError(
            f"No usable quarterly financial data found for {ticker}. "
            f"SEC diagnostics: {sec_diag[-2:]}; Yahoo diagnostics: {yf_diag[-2:]}"
        )

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("Period")
        .drop_duplicates("Period", keep="last")
        .reset_index(drop=True)
    )

    implied_shares = safe_divide(df["Net_Income"], df["Diluted_EPS"])
    missing_shares = df["Diluted_Shares"].isna()
    df.loc[missing_shares, "Diluted_Shares"] = implied_shares[missing_shares]

    df["Revenue_B"] = df["Revenue"] / 1e9
    df["Rev_YoY_%"] = pct_change_safe(df["Revenue_B"], 4)
    df["Rev_QoQ_%"] = pct_change_safe(df["Revenue_B"], 1)

    df["EPS_YoY_%"] = pct_change_safe(df["Diluted_EPS"], 4).clip(-200, 200)
    df["EPS_QoQ_%"] = pct_change_safe(df["Diluted_EPS"], 1).clip(-200, 200)

    df["Op_Margin_%"] = safe_divide(df["Operating_Income"], df["Revenue"]) * 100

    df["Net_Margin_%"] = safe_divide(df["Net_Income"], df["Revenue"]) * 100

    df["Capex_B"] = np.abs(df["Capex"]) / 1e9
    df["Diluted_Shares_M"] = df["Diluted_Shares"] / 1e6

    df["Share_Dilution_YoY_%"] = pct_change_safe(
        df["Diluted_Shares_M"], 4
    ).clip(-25, 25)

    fcf_df = calculate_fcf_safe(df)

    return df, fcf_df, sec_diag, yf_diag


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_and_shares(ticker):
    try:
        tk = yf.Ticker(ticker)

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        market_cap = info.get("marketCap")
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        hist = tk.history(period="2y", auto_adjust=False)

        if not hist.empty and "Close" in hist.columns:
            hist = hist.copy()
            hist["EMA50"] = hist["Close"].ewm(
                span=50, adjust=False
            ).mean()
            hist["EMA200"] = hist["Close"].ewm(
                span=200, adjust=False
            ).mean()

            if not current_price:
                current_price = hist["Close"].iloc[-1]

        shares_outstanding = info.get("sharesOutstanding")

        if not market_cap and current_price and shares_outstanding:
            market_cap = current_price * shares_outstanding

        return hist, current_price, market_cap, shares_outstanding

    except Exception:
        return pd.DataFrame(), None, None, None


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

if run_button or ticker_symbol:
    try:
        with st.spinner(
            f"Extracting SEC/XBRL and market data for {ticker_symbol}..."
        ):
            (
                df_raw_full,
                df_fcf_full,
                sec_diag,
                yf_diag,
            ) = fetch_and_parse_ticker(ticker_symbol)

            (
                hist_price,
                current_price,
                market_cap,
                shares_outstanding,
            ) = fetch_market_and_shares(ticker_symbol)

        df_raw = df_raw_full.tail(lookback_quarters).reset_index(drop=True)
        df_fcf = df_fcf_full.tail(lookback_quarters).reset_index(drop=True)

        if market_cap and current_price:
            df_raw["TTM_Revenue"] = df_raw["Revenue"].rolling(4).sum()
            df_raw["Ann_Revenue"] = df_raw["Revenue"] * 4

            df_raw["P_S_TTM"] = (
                market_cap / df_raw["TTM_Revenue"]
            ).where(df_raw["TTM_Revenue"] > 0).clip(lower=0, upper=150)

            df_raw["P_S_Ann"] = (
                market_cap / df_raw["Ann_Revenue"]
            ).where(df_raw["Ann_Revenue"] > 0).clip(lower=0, upper=150)

            df_raw["TTM_EPS"] = df_raw["Diluted_EPS"].rolling(4).sum()
            df_raw["Ann_EPS"] = df_raw["Diluted_EPS"] * 4

            df_raw["P_E_TTM"] = (
                current_price / df_raw["TTM_EPS"]
            ).where(df_raw["TTM_EPS"] > 0).clip(lower=0, upper=200)

            df_raw["P_E_Ann"] = (
                current_price / df_raw["Ann_EPS"]
            ).where(df_raw["Ann_EPS"] > 0).clip(lower=0, upper=200)

            merged_fcf = df_raw[["Period"]].merge(
                df_fcf[["Period", "FCF"]],
                on="Period",
                how="left",
            )

            df_raw["TTM_FCF"] = merged_fcf["FCF"].rolling(4).sum()

            df_raw["FCF_Yield_%"] = (df_raw["TTM_FCF"] / market_cap) * 100

        st.subheader(
            f"{ticker_symbol} — Executive Financial Dashboard"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Current Price",
                f"${current_price:,.2f}" if current_price else "N/A",
            )

        with c2:
            st.metric(
                "Market Cap",
                f"${market_cap/1e9:,.2f}B" if market_cap else "N/A",
            )

        with c3:
            st.metric(
                "Quarterly Records",
                f"{len(df_raw_full)}",
            )

        with c4:
            source_counts = df_raw_full["Source"].value_counts()
            st.metric(
                "SEC Records",
                f"{source_counts.get('SEC XBRL', 0) + source_counts.get('SEC + Yahoo', 0)}",
            )

        with st.expander("Data Source Diagnostics"):
            st.write("**Sources used:**")
            st.dataframe(
                df_raw_full[
                    ["Period", "Form", "Source"]
                ].tail(20),
                use_container_width=True,
            )

            if sec_diag:
                st.write("**SEC diagnostics (most recent):**")
                for msg in sec_diag[-10:]:
                    st.caption(msg)

            if yf_diag:
                st.write("**Yahoo diagnostics:**")
                for msg in yf_diag[-10:]:
                    st.caption(msg)

        fig, axes = plt.subplots(
            2, 2, figsize=(16, 11), dpi=150
        )

        fig.suptitle(
            f"{ticker_symbol} Financial & Growth Dashboard",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Revenue
        ax1 = axes[0, 0]
        v_rev = df_raw.dropna(subset=["Revenue_B"])

        if not v_rev.empty:
            ax1.bar(
                v_rev["Period"].astype(str),
                v_rev["Revenue_B"],
                alpha=0.85,
                width=0.55,
                label="Revenue ($B)",
            )

            ax1.set_title(
                "Revenue ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax1.set_ylabel("Revenue ($ Billions)")
            ax1.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax1_sub = ax1.twinx()

            ax1_sub.plot(
                v_rev["Period"].astype(str),
                v_rev["Rev_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax1_sub.plot(
                v_rev["Period"].astype(str),
                v_rev["Rev_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax1_sub.set_ylabel("Growth (%)")
            ax1_sub.grid(False)

            l1, lb1 = ax1.get_legend_handles_labels()
            l1s, lb1s = ax1_sub.get_legend_handles_labels()

            ax1.legend(
                l1 + l1s,
                lb1 + lb1s,
                loc="upper left",
                fontsize=6.5,
            )

        # EPS
        ax2 = axes[0, 1]
        v_eps = df_raw.dropna(subset=["Diluted_EPS"])

        if not v_eps.empty:
            ax2.bar(
                v_eps["Period"].astype(str),
                v_eps["Diluted_EPS"],
                alpha=0.85,
                width=0.55,
                label="Diluted EPS ($)",
            )

            ax2.set_title(
                "Diluted EPS ($) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax2.set_ylabel("EPS ($)")
            ax2.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax2_sub = ax2.twinx()

            ax2_sub.plot(
                v_eps["Period"].astype(str),
                v_eps["EPS_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax2_sub.plot(
                v_eps["Period"].astype(str),
                v_eps["EPS_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax2_sub.set_ylabel("Growth (%)")
            ax2_sub.grid(False)

            l2, lb2 = ax2.get_legend_handles_labels()
            l2s, lb2s = ax2_sub.get_legend_handles_labels()

            ax2.legend(
                l2 + l2s,
                lb2 + lb2s,
                loc="upper left",
                fontsize=6.5,
            )

        # FCF
        ax3 = axes[1, 0]
        v_fcf = df_fcf.tail(lookback_quarters).copy()
        v_fcf_clean = v_fcf.dropna(subset=["FCF_B"])

        if not v_fcf_clean.empty:
            ax3.bar(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_B"],
                alpha=0.85,
                width=0.55,
                label="Free Cash Flow ($B)",
            )

            ax3.set_title(
                "Standalone Free Cash Flow ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax3.set_ylabel("FCF ($ Billions)")
            ax3.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax3_sub = ax3.twinx()

            ax3_sub.plot(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax3_sub.plot(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax3_sub.set_ylabel("Growth (%)")
            ax3_sub.grid(False)

            l3, lb3 = ax3.get_legend_handles_labels()
            l3s, lb3s = ax3_sub.get_legend_handles_labels()

            ax3.legend(
                l3 + l3s,
                lb3 + lb3s,
                loc="upper left",
                fontsize=6.5,
            )

        # Margins
        ax4 = axes[1, 1]

        ax4.plot(
            df_raw["Period"].astype(str),
            df_raw["Op_Margin_%"],
            marker="o",
            linewidth=2,
            label="Operating Margin (%)",
        )

        ax4.plot(
            df_raw["Period"].astype(str),
            df_raw["Net_Margin_%"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Net Margin (%)",
        )

        ax4.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax4.set_title(
            "Operating Margin vs. Net Margin (%)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax4.set_ylabel("Margin (%)")
        ax4.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax4.legend(
            loc="upper left",
            fontsize=7,
        )
        ax4.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )
        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------------------------
        # VALUATION
        # -------------------------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Valuation Multiples & Price Action"
        )

        fig2, axes2 = plt.subplots(
            2, 2, figsize=(16, 11), dpi=150
        )

        fig2.suptitle(
            f"{ticker_symbol} Price Action, TTM/Annualized Multiples & FCF Yield",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Price / EMA
        ax_p1 = axes2[0, 0]

        if hist_price is not None and not hist_price.empty:
            ax_p1.plot(
                hist_price.index,
                hist_price["Close"],
                linewidth=1.5,
                label="Close Price ($)",
            )

            if "EMA50" in hist_price.columns:
                ax_p1.plot(
                    hist_price.index,
                    hist_price["EMA50"],
                    linestyle="-",
                    linewidth=1.2,
                    label="50-Day EMA",
                )

            if "EMA200" in hist_price.columns:
                ax_p1.plot(
                    hist_price.index,
                    hist_price["EMA200"],
                    linestyle="--",
                    linewidth=1.2,
                    label="200-Day EMA",
                )

        ax_p1.set_title(
            "Daily Stock Price vs 50/200 EMA",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p1.set_ylabel("Price ($)")
        ax_p1.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p1.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p1.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/S
        ax_p2 = axes2[0, 1]

        if "P_S_TTM" in df_raw.columns:
            ax_p2.plot(
                df_raw["Period"].astype(str),
                df_raw["P_S_TTM"],
                marker="o",
                linewidth=2,
                label="P/S (TTM)",
            )

            ax_p2.plot(
                df_raw["Period"].astype(str),
                df_raw["P_S_Ann"],
                marker="^",
                linestyle="--",
                linewidth=1.5,
                label="P/S (Annualized Quarter)",
            )

        ax_p2.set_title(
            "Price-to-Sales (P/S): TTM vs Annualized Q",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p2.set_ylabel("P/S Multiple (x)")
        ax_p2.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p2.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p2.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/E
        ax_p3 = axes2[1, 0]

        if "P_E_TTM" in df_raw.columns:
            ax_p3.plot(
                df_raw["Period"].astype(str),
                df_raw["P_E_TTM"],
                marker="s",
                linewidth=2,
                label="P/E (TTM)",
            )

            ax_p3.plot(
                df_raw["Period"].astype(str),
                df_raw["P_E_Ann"],
                marker="d",
                linestyle="--",
                linewidth=1.5,
                label="P/E (Annualized Quarter)",
            )

        ax_p3.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_p3.set_title(
            "Price-to-Earnings (P/E): TTM vs Annualized Q",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p3.set_ylabel("P/E Multiple (x)")
        ax_p3.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p3.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p3.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # FCF Yield
        ax_p4 = axes2[1, 1]

        if "FCF_Yield_%" in df_raw.columns:
            ax_p4.plot(
                df_raw["Period"].astype(str),
                df_raw["FCF_Yield_%"],
                marker="^",
                linewidth=2,
                label="FCF Yield (TTM %)",
            )

        ax_p4.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_p4.set_title(
            "Free Cash Flow Yield (TTM %)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p4.set_ylabel("FCF Yield (%)")
        ax_p4.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p4.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p4.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )
        st.pyplot(fig2)
        plt.close(fig2)

        # -------------------------------------------------------------------
        # CAPEX / DILUTION
        -- -------------------------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Capital Expenditures & Share Dilution"
        )

        fig3, axes3 = plt.subplots(
            1, 2, figsize=(16, 5), dpi=150
        )

        ax_c1 = axes3[0]

        ax_c1.bar(
            df_raw["Period"].astype(str),
            df_raw["Capex_B"],
            alpha=0.85,
            width=0.55,
            label="Capex ($B)",
        )

        ax_c1.set_title(
            "Quarterly Capital Expenditures ($B)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_c1.set_ylabel("Capex ($ Billions)")
        ax_c1.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_c1.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_c1.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        ax_c2 = axes3[1]

        ax_c2.plot(
            df_raw["Period"].astype(str),
            df_raw["Share_Dilution_YoY_%"],
            marker="o",
            linewidth=2,
            label="Diluted Share Growth YoY (%)",
        )

        ax_c2.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_c2.set_title(
            "Share Dilution / Buyback Rate (YoY % Change)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_c2.set_ylabel("Share Count YoY Change (%)")
        ax_c2.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_c2.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_c2.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # -------------------------------------------------------------------
        # RAW DATA
        # -------------------------------------------------------------------

        with st.expander(
            "View Raw Extracted Dataset & Valuations"
        ):
            display_df = df_raw.copy()
            display_df["Period"] = display_df["Period"].dt.strftime("%Y-%m-%d")

            st.dataframe(
                display_df,
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            f"Could not load data for ticker '{ticker_symbol}'. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        st.exception(exc)        "**Data hierarchy:** SEC XBRL/EdgarTools first; Yahoo Finance is used "
        "as a fallback for financial data and for market-price history."
    )
    st.caption(
        "**Audit advisory:** Cross-reference important XBRL values against "
        "the company's official SEC filing before making investment decisions."
    )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def clean_number(value):
    """Convert common SEC/Yahoo numeric representations to float."""
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if pd.notna(value) else np.nan

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


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def is_reasonable_numeric(x):
    return pd.notna(x) and np.isfinite(x)


def pct_change_safe(series, periods):
    return series.pct_change(periods=periods, fill_method=None) * 100


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


# ---------------------------------------------------------------------------
# XBRL / SEC EXTRACTION
# ---------------------------------------------------------------------------

CONCEPT_CANDIDATES = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Revenue",
    ],
    "Operating_Income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "Net_Income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "Diluted_EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "BasicAndDilutedEarningsPerShare",
    ],
    "Diluted_Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ],
    "OCF": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "Capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForProceedsFromOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}


def find_statement_value(statement_df, candidates):
    """Find a statement value using standardized XBRL concept/label information."""
    if statement_df is None or statement_df.empty:
        return np.nan

    df = statement_df.copy()

    concept_col = None
    for c in ["standard_concept", "concept", "label", "name"]:
        if c in df.columns:
            concept_col = c
            break

    if concept_col is None:
        working = df.copy()
        working["_row_text"] = working.index.astype(str)
    else:
        working = df.copy()
        working["_row_text"] = working[concept_col].astype(str)

    for candidate in candidates:
        candidate_norm = normalize_label(candidate)

        exact = working[
            working["_row_text"].map(normalize_label) == candidate_norm
        ]
        if not exact.empty:
            row = exact.iloc[0]
            value = _latest_numeric_period_value(row)
            if is_reasonable_numeric(value):
                return value

    candidate_words = [normalize_label(x) for x in candidates]
    mask = working["_row_text"].map(
        lambda x: any(cw in normalize_label(x) for cw in candidate_words)
    )
    matches = working[mask]

    if len(matches) == 1:
        return _latest_numeric_period_value(matches.iloc[0])

    return np.nan


def _latest_numeric_period_value(row):
    """Select a numeric value from a statement row."""
    candidates = []

    for col in row.index:
        if str(col).startswith("_"):
            continue

        value = clean_number(row[col])
        if not is_reasonable_numeric(value):
            continue

        text = str(col)
        if any(
            x in text.lower()
            for x in ["label", "concept", "standard_concept", "units"]
        ):
            continue

        score = 0
        upper = text.upper()

        if "Q" in upper:
            score += 10
        if "FY" in upper:
            score += 3
        if "202" in upper:
            score += 2

        candidates.append((score, text, value))

    if not candidates:
        return np.nan

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _period_from_filing(filing):
    """Get filing report date safely."""
    try:
        return pd.to_datetime(str(filing.period_of_report)[:10], errors="coerce")
    except Exception:
        return pd.NaT


def _extract_from_edgar_statements(ticker):
    """Primary SEC path."""
    records = []
    diagnostics = []

    try:
        company = Company(ticker)
        filings = company.get_filings(form=["10-Q", "10-K", "20-F"])
        filings = list(filings)[:60]

        for filing in filings:
            period = _period_from_filing(filing)
            if pd.isna(period):
                continue

            form = str(getattr(filing, "form", ""))

            try:
                obj = filing.obj()

                inc = getattr(obj, "income_statement", None)
                cf = getattr(obj, "cash_flow_statement", None)

                inc_df = (
                    inc.to_dataframe(view="standard")
                    if inc is not None
                    else None
                )
                cf_df = (
                    cf.to_dataframe(view="standard") if cf is not None else None
                )

                if inc_df is None or inc_df.empty:
                    continue

                record = {
                    "Period": period,
                    "Form": form,
                    "Source": "SEC XBRL",
                    "Revenue": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Revenue"]
                    ),
                    "Operating_Income": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Operating_Income"]
                    ),
                    "Net_Income": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Net_Income"]
                    ),
                    "Diluted_EPS": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Diluted_EPS"]
                    ),
                    "Diluted_Shares": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Diluted_Shares"]
                    ),
                    "OCF": find_statement_value(
                        cf_df, CONCEPT_CANDIDATES["OCF"]
                    ),
                    "Capex": find_statement_value(
                        cf_df, CONCEPT_CANDIDATES["Capex"]
                    ),
                }

                if is_reasonable_numeric(record["Revenue"]) or is_reasonable_numeric(
                    record["Net_Income"]
                ):
                    records.append(record)

            except Exception as exc:
                diagnostics.append(
                    f"{period.date()} {form}: {type(exc).__name__}: {exc}"
                )
                continue

    except Exception as exc:
        diagnostics.append(
            f"SEC company/fillings error: {type(exc).__name__}: {exc}"
        )

    if not records:
        return pd.DataFrame(), diagnostics

    df = pd.DataFrame(records)
    df["completeness"] = df.notna().sum(axis=1)

    df = (
        df.sort_values(["Period", "completeness"], ascending=[True, False])
        .drop_duplicates("Period", keep="first")
        .drop(columns=["completeness"])
        .reset_index(drop=True)
    )

    return df, diagnostics


# ---------------------------------------------------------------------------
# YAHOO FALLBACK / SUPPLEMENT
# ---------------------------------------------------------------------------

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
        "Basic EPS",
    ],
    "Diluted_Shares": [
        "Diluted Average Shares",
        "Diluted Average Shares Outstanding",
        "Basic Average Shares",
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
                value = df.loc[key, date_col]
                return clean_number(value)
            except Exception:
                continue
    return np.nan


def fetch_yahoo_quarterly(ticker):
    """Yahoo fallback."""
    diagnostics = []

    try:
        tk = yf.Ticker(ticker)

        inc = tk.quarterly_income_stmt
        cf = tk.quarterly_cashflow

        if inc is None or inc.empty:
            return pd.DataFrame(), ["Yahoo quarterly income statement empty"]

        records = []

        for date_col in inc.columns:
            period = pd.to_datetime(date_col, errors="coerce")
            if pd.isna(period):
                continue

            records.append(
                {
                    "Period": period,
                    "Form": "Yahoo",
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
                    "OCF": yahoo_row_value(cf, YF_KEYS["OCF"], date_col),
                    "Capex": yahoo_row_value(cf, YF_KEYS["Capex"], date_col),
                }
            )

        return pd.DataFrame(records), diagnostics

    except Exception as exc:
        diagnostics.append(f"Yahoo error: {type(exc).__name__}: {exc}")
        return pd.DataFrame(), diagnostics


def merge_sec_and_yahoo(sec_df, yf_df):
    """SEC is authoritative where a value exists. Yahoo fills missing fields."""
    if sec_df.empty:
        return yf_df.copy()

    if yf_df.empty:
        return sec_df.copy()

    all_cols = [
        "Period",
        "Form",
        "Source",
        "Revenue",
        "Operating_Income",
        "Net_Income",
        "Diluted_EPS",
        "Diluted_Shares",
        "OCF",
        "Capex",
    ]

    sec = sec_df.copy()
    yahoo = yf_df.copy()

    for df in [sec, yahoo]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan

    combined = pd.concat(
        [sec[all_cols], yahoo[all_cols]],
        ignore_index=True,
    )

    combined["source_priority"] = np.where(
        combined["Source"].eq("SEC XBRL"), 0, 1
    )

    combined = combined.sort_values(["Period", "source_priority"])

    output = []

    for period, group in combined.groupby("Period", sort=True):
        row = {
            "Period": period,
            "Form": group.iloc[0]["Form"],
            "Source": group.iloc[0]["Source"],
        }

        for col in [
            "Revenue",
            "Operating_Income",
            "Net_Income",
            "Diluted_EPS",
            "Diluted_Shares",
            "OCF",
            "Capex",
        ]:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[col] = values.iloc[0] if not values.empty else np.nan

        if len(group) > 1:
            sec_row = group[group["Source"].eq("SEC XBRL")]
            yf_row = group[group["Source"].eq("Yahoo Finance")]

            if not sec_row.empty and not yf_row.empty:
                for col in [
                    "Revenue",
                    "Operating_Income",
                    "Net_Income",
                    "Diluted_EPS",
                    "Diluted_Shares",
                    "OCF",
                    "Capex",
                ]:
                    sec_value = pd.to_numeric(
                        sec_row.iloc[0][col], errors="coerce"
                    )
                    yf_value = pd.to_numeric(
                        yf_row.iloc[0][col], errors="coerce"
                    )

                    if pd.isna(sec_value) and pd.notna(yf_value):
                        row[col] = yf_value
                        row["Source"] = "SEC + Yahoo"

        output.append(row)

    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# CASH FLOW NORMALIZATION
# ---------------------------------------------------------------------------


def calculate_fcf_safe(df):
    """Calculate standalone FCF."""
    if df.empty:
        return pd.DataFrame(columns=["Period", "FCF", "FCF_B"])

    work = df[["Period", "OCF", "Capex"]].copy()
    work["Period"] = pd.to_datetime(work["Period"], errors="coerce")
    work["OCF"] = pd.to_numeric(work["OCF"], errors="coerce")
    work["Capex"] = pd.to_numeric(work["Capex"], errors="coerce")
    work = work.sort_values("Period").reset_index(drop=True)

    standalone_ocf = []
    standalone_capex = []

    previous_year = None
    previous_ocf = np.nan
    previous_capex = np.nan

    for _, row in work.iterrows():
        period = row["Period"]
        year = period.year if pd.notna(period) else None

        ocf = row["OCF"]
        capex = row["Capex"]

        q_ocf = ocf
        q_capex = capex

        if year == previous_year:
            if pd.notna(ocf) and pd.notna(previous_ocf):
                if abs(ocf) >= abs(previous_ocf) * 1.10:
                    q_ocf = ocf - previous_ocf

            if pd.notna(capex) and pd.notna(previous_capex):
                if abs(capex) >= abs(previous_capex) * 1.10:
                    q_capex = capex - previous_capex

        standalone_ocf.append(q_ocf)
        standalone_capex.append(q_capex)

        previous_year = year
        previous_ocf = ocf
        previous_capex = capex

    fcf = np.array(standalone_ocf, dtype=float) - np.abs(
        np.array(standalone_capex, dtype=float)
    )

    out = pd.DataFrame(
        {
            "Period": work["Period"],
            "FCF": fcf,
        }
    )

    out["FCF_B"] = out["FCF"] / 1e9
    out["FCF_YoY_%"] = pct_change_safe(out["FCF"], 4).clip(-200, 200)
    out["FCF_QoQ_%"] = pct_change_safe(out["FCF"], 1).clip(-200, 200)

    return out


# ---------------------------------------------------------------------------
# MAIN DATA FETCH
# ---------------------------------------------------------------------------


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_and_parse_ticker(ticker):
    sec_df, sec_diag = _extract_from_edgar_statements(ticker)
    yf_df, yf_diag = fetch_yahoo_quarterly(ticker)

    df = merge_sec_and_yahoo(sec_df, yf_df)

    if df.empty:
        raise ValueError(
            f"No usable quarterly financial data found for {ticker}. "
            f"SEC diagnostics: {sec_diag[-2:]}; Yahoo diagnostics: {yf_diag[-2:]}"
        )

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("Period")
        .drop_duplicates("Period", keep="last")
        .reset_index(drop=True)
    )

    implied_shares = safe_divide(df["Net_Income"], df["Diluted_EPS"])
    missing_shares = df["Diluted_Shares"].isna()
    df.loc[missing_shares, "Diluted_Shares"] = implied_shares[missing_shares]

    df["Revenue_B"] = df["Revenue"] / 1e9
    df["Rev_YoY_%"] = pct_change_safe(df["Revenue_B"], 4)
    df["Rev_QoQ_%"] = pct_change_safe(df["Revenue_B"], 1)

    df["EPS_YoY_%"] = pct_change_safe(df["Diluted_EPS"], 4).clip(-200, 200)
    df["EPS_QoQ_%"] = pct_change_safe(df["Diluted_EPS"], 1).clip(-200, 200)

    df["Op_Margin_%"] = safe_divide(df["Operating_Income"], df["Revenue"]) * 100

    df["Net_Margin_%"] = safe_divide(df["Net_Income"], df["Revenue"]) * 100

    df["Capex_B"] = np.abs(df["Capex"]) / 1e9
    df["Diluted_Shares_M"] = df["Diluted_Shares"] / 1e6

    df["Share_Dilution_YoY_%"] = pct_change_safe(
        df["Diluted_Shares_M"], 4
    ).clip(-25, 25)

    fcf_df = calculate_fcf_safe(df)

    return df, fcf_df, sec_diag, yf_diag


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_and_shares(ticker):
    try:
        tk = yf.Ticker(ticker)

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        market_cap = info.get("marketCap")
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        hist = tk.history(period="2y", auto_adjust=False)

        if not hist.empty and "Close" in hist.columns:
            hist = hist.copy()
            hist["EMA50"] = hist["Close"].ewm(
                span=50, adjust=False
            ).mean()
            hist["EMA200"] = hist["Close"].ewm(
                span=200, adjust=False
            ).mean()

            if not current_price:
                current_price = hist["Close"].iloc[-1]

        shares_outstanding = info.get("sharesOutstanding")

        if not market_cap and current_price and shares_outstanding:
            market_cap = current_price * shares_outstanding

        return hist, current_price, market_cap, shares_outstanding

    except Exception:
        return pd.DataFrame(), None, None, None


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

if run_button or ticker_symbol:
    try:
        with st.spinner(
            f"Extracting SEC/XBRL and market data for {ticker_symbol}..."
        ):
            (
                df_raw_full,
                df_fcf_full,
                sec_diag,
                yf_diag,
            ) = fetch_and_parse_ticker(ticker_symbol)

            (
                hist_price,
                current_price,
                market_cap,
                shares_outstanding,
            ) = fetch_market_and_shares(ticker_symbol)

        df_raw = df_raw_full.tail(lookback_quarters).reset_index(drop=True)
        df_fcf = df_fcf_full.tail(lookback_quarters).reset_index(drop=True)

        if market_cap and current_price:
            df_raw["TTM_Revenue"] = df_raw["Revenue"].rolling(4).sum()
            df_raw["Ann_Revenue"] = df_raw["Revenue"] * 4

            df_raw["P_S_TTM"] = (
                market_cap / df_raw["TTM_Revenue"]
            ).where(df_raw["TTM_Revenue"] > 0).clip(lower=0, upper=150)

            df_raw["P_S_Ann"] = (
                market_cap / df_raw["Ann_Revenue"]
            ).where(df_raw["Ann_Revenue"] > 0).clip(lower=0, upper=150)

            df_raw["TTM_EPS"] = df_raw["Diluted_EPS"].rolling(4).sum()
            df_raw["Ann_EPS"] = df_raw["Diluted_EPS"] * 4

            df_raw["P_E_TTM"] = (
                current_price / df_raw["TTM_EPS"]
            ).where(df_raw["TTM_EPS"] > 0).clip(lower=0, upper=200)

            df_raw["P_E_Ann"] = (
                current_price / df_raw["Ann_EPS"]
            ).where(df_raw["Ann_EPS"] > 0).clip(lower=0, upper=200)

            merged_fcf = df_raw[["Period"]].merge(
                df_fcf[["Period", "FCF"]],
                on="Period",
                how="left",
            )

            df_raw["TTM_FCF"] = merged_fcf["FCF"].rolling(4).sum()

            df_raw["FCF_Yield_%"] = (df_raw["TTM_FCF"] / market_cap) * 100

        st.subheader(
            f"{ticker_symbol} — Executive Financial Dashboard"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Current Price",
                f"${current_price:,.2f}" if current_price else "N/A",
            )

        with c2:
            st.metric(
                "Market Cap",
                f"${market_cap/1e9:,.2f}B" if market_cap else "N/A",
            )

        with c3:
            st.metric(
                "Quarterly Records",
                f"{len(df_raw_full)}",
            )

        with c4:
            source_counts = df_raw_full["Source"].value_counts()
            st.metric(
                "SEC Records",
                f"{source_counts.get('SEC XBRL', 0) + source_counts.get('SEC + Yahoo', 0)}",
            )

        with st.expander("Data Source Diagnostics"):
            st.write("**Sources used:**")
            st.dataframe(
                df_raw_full[
                    ["Period", "Form", "Source"]
                ].tail(20),
                use_container_width=True,
            )

            if sec_diag:
                st.write("**SEC diagnostics (most recent):**")
                for msg in sec_diag[-10:]:
                    st.caption(msg)

            if yf_diag:
                st.write("**Yahoo diagnostics:**")
                for msg in yf_diag[-10:]:
                    st.caption(msg)

        fig, axes = plt.subplots(
            2, 2, figsize=(16, 11), dpi=150
        )

        fig.suptitle(
            f"{ticker_symbol} Financial & Growth Dashboard",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Revenue
        ax1 = axes[0, 0]
        v_rev = df_raw.dropna(subset=["Revenue_B"])

        if not v_rev.empty:
            ax1.bar(
                v_rev["Period"].astype(str),
                v_rev["Revenue_B"],
                alpha=0.85,
                width=0.55,
                label="Revenue ($B)",
            )

            ax1.set_title(
                "Revenue ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax1.set_ylabel("Revenue ($ Billions)")
            ax1.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax1_sub = ax1.twinx()

            ax1_sub.plot(
                v_rev["Period"].astype(str),
                v_rev["Rev_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax1_sub.plot(
                v_rev["Period"].astype(str),
                v_rev["Rev_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax1_sub.set_ylabel("Growth (%)")
            ax1_sub.grid(False)

            l1, lb1 = ax1.get_legend_handles_labels()
            l1s, lb1s = ax1_sub.get_legend_handles_labels()

            ax1.legend(
                l1 + l1s,
                lb1 + lb1s,
                loc="upper left",
                fontsize=6.5,
            )

        # EPS
        ax2 = axes[0, 1]
        v_eps = df_raw.dropna(subset=["Diluted_EPS"])

        if not v_eps.empty:
            ax2.bar(
                v_eps["Period"].astype(str),
                v_eps["Diluted_EPS"],
                alpha=0.85,
                width=0.55,
                label="Diluted EPS ($)",
            )

            ax2.set_title(
                "Diluted EPS ($) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax2.set_ylabel("EPS ($)")
            ax2.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax2_sub = ax2.twinx()

            ax2_sub.plot(
                v_eps["Period"].astype(str),
                v_eps["EPS_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax2_sub.plot(
                v_eps["Period"].astype(str),
                v_eps["EPS_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax2_sub.set_ylabel("Growth (%)")
            ax2_sub.grid(False)

            l2, lb2 = ax2.get_legend_handles_labels()
            l2s, lb2s = ax2_sub.get_legend_handles_labels()

            ax2.legend(
                l2 + l2s,
                lb2 + lb2s,
                loc="upper left",
                fontsize=6.5,
            )

        # FCF
        ax3 = axes[1, 0]
        v_fcf = df_fcf.tail(lookback_quarters).copy()
        v_fcf_clean = v_fcf.dropna(subset=["FCF_B"])

        if not v_fcf_clean.empty:
            ax3.bar(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_B"],
                alpha=0.85,
                width=0.55,
                label="Free Cash Flow ($B)",
            )

            ax3.set_title(
                "Standalone Free Cash Flow ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax3.set_ylabel("FCF ($ Billions)")
            ax3.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax3_sub = ax3.twinx()

            ax3_sub.plot(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax3_sub.plot(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax3_sub.set_ylabel("Growth (%)")
            ax3_sub.grid(False)

            l3, lb3 = ax3.get_legend_handles_labels()
            l3s, lb3s = ax3_sub.get_legend_handles_labels()

            ax3.legend(
                l3 + l3s,
                lb3 + lb3s,
                loc="upper left",
                fontsize=6.5,
            )

        # Margins
        ax4 = axes[1, 1]

        ax4.plot(
            df_raw["Period"].astype(str),
            df_raw["Op_Margin_%"],
            marker="o",
            linewidth=2,
            label="Operating Margin (%)",
        )

        ax4.plot(
            df_raw["Period"].astype(str),
            df_raw["Net_Margin_%"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Net Margin (%)",
        )

        ax4.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax4.set_title(
            "Operating Margin vs. Net Margin (%)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax4.set_ylabel("Margin (%)")
        ax4.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax4.legend(
            loc="upper left",
            fontsize=7,
        )
        ax4.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )
        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------------------------
        # VALUATION
        # -------------------------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Valuation Multiples & Price Action"
        )

        fig2, axes2 = plt.subplots(
            2, 2, figsize=(16, 11), dpi=150
        )

        fig2.suptitle(
            f"{ticker_symbol} Price Action, TTM/Annualized Multiples & FCF Yield",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Price / EMA
        ax_p1 = axes2[0, 0]

        if hist_price is not None and not hist_price.empty:
            ax_p1.plot(
                hist_price.index,
                hist_price["Close"],
                linewidth=1.5,
                label="Close Price ($)",
            )

            if "EMA50" in hist_price.columns:
                ax_p1.plot(
                    hist_price.index,
                    hist_price["EMA50"],
                    linestyle="-",
                    linewidth=1.2,
                    label="50-Day EMA",
                )

            if "EMA200" in hist_price.columns:
                ax_p1.plot(
                    hist_price.index,
                    hist_price["EMA200"],
                    linestyle="--",
                    linewidth=1.2,
                    label="200-Day EMA",
                )

        ax_p1.set_title(
            "Daily Stock Price vs 50/200 EMA",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p1.set_ylabel("Price ($)")
        ax_p1.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p1.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p1.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/S
        ax_p2 = axes2[0, 1]

        if "P_S_TTM" in df_raw.columns:
            ax_p2.plot(
                df_raw["Period"].astype(str),
                df_raw["P_S_TTM"],
                marker="o",
                linewidth=2,
                label="P/S (TTM)",
            )

            ax_p2.plot(
                df_raw["Period"].astype(str),
                df_raw["P_S_Ann"],
                marker="^",
                linestyle="--",
                linewidth=1.5,
                label="P/S (Annualized Quarter)",
            )

        ax_p2.set_title(
            "Price-to-Sales (P/S): TTM vs Annualized Q",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p2.set_ylabel("P/S Multiple (x)")
        ax_p2.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p2.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p2.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/E
        ax_p3 = axes2[1, 0]

        if "P_E_TTM" in df_raw.columns:
            ax_p3.plot(
                df_raw["Period"].astype(str),
                df_raw["P_E_TTM"],
                marker="s",
                linewidth=2,
                label="P/E (TTM)",
            )

            ax_p3.plot(
                df_raw["Period"].astype(str),
                df_raw["P_E_Ann"],
                marker="d",
                linestyle="--",
                linewidth=1.5,
                label="P/E (Annualized Quarter)",
            )

        ax_p3.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_p3.set_title(
            "Price-to-Earnings (P/E): TTM vs Annualized Q",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p3.set_ylabel("P/E Multiple (x)")
        ax_p3.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p3.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p3.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # FCF Yield
        ax_p4 = axes2[1, 1]

        if "FCF_Yield_%" in df_raw.columns:
            ax_p4.plot(
                df_raw["Period"].astype(str),
                df_raw["FCF_Yield_%"],
                marker="^",
                linewidth=2,
                label="FCF Yield (TTM %)",
            )

        ax_p4.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_p4.set_title(
            "Free Cash Flow Yield (TTM %)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p4.set_ylabel("FCF Yield (%)")
        ax_p4.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p4.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p4.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )
        st.pyplot(fig2)
        plt.close(fig2)

        # -------------------------------------------------------------------
        # CAPEX / DILUTION
        # -------------------------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Capital Expenditures & Share Dilution"
        )

        fig3, axes3 = plt.subplots(
            1, 2, figsize=(16, 5), dpi=150
        )

        ax_c1 = axes3[0]

        ax_c1.bar(
            df_raw["Period"].astype(str),
            df_raw["Capex_B"],
            alpha=0.85,
            width=0.55,
            label="Capex ($B)",
        )

        ax_c1.set_title(
            "Quarterly Capital Expenditures ($B)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_c1.set_ylabel("Capex ($ Billions)")
        ax_c1.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_c1.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_c1.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        ax_c2 = axes3[1]

        ax_c2.plot(
            df_raw["Period"].astype(str),
            df_raw["Share_Dilution_YoY_%"],
            marker="o",
            linewidth=2,
            label="Diluted Share Growth YoY (%)",
        )

        ax_c2.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_c2.set_title(
            "Share Dilution / Buyback Rate (YoY % Change)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_c2.set_ylabel("Share Count YoY Change (%)")
        ax_c2.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_c2.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_c2.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # -------------------------------------------------------------------
        # RAW DATA
        # -------------------------------------------------------------------

        with st.expander(
            "View Raw Extracted Dataset & Valuations"
        ):
            display_df = df_raw.copy()
            display_df["Period"] = display_df["Period"].dt.strftime("%Y-%m-%d")

            st.dataframe(
                display_df,
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            f"Could not load data for ticker '{ticker_symbol}. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        st.exception(exc)
    st.markdown("---")
    st.caption(
        "**Data hierarchy:** SEC XBRL/EdgarTools first; Yahoo Finance is used "
        "as a fallback for financial data and for market-price history."
    )
    st.caption(
        "**Audit advisory:** Cross-reference important XBRL values against "
        "the company's official SEC filing before making investment decisions."
    )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def clean_number(value):
    """Convert common SEC/Yahoo numeric representations to float."""
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if pd.notna(value) else np.nan

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


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def is_reasonable_numeric(x):
    return pd.notna(x) and np.isfinite(x)


def pct_change_safe(series, periods):
    return series.pct_change(periods=periods, fill_method=None) * 100


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


# ---------------------------------------------------------------------------
# XBRL / SEC EXTRACTION
# ---------------------------------------------------------------------------

# These are deliberately prioritized rather than using a broad "first label
# containing Revenue" search. This avoids accidentally selecting a subtotal,
# segment, or comparative-period row.
CONCEPT_CANDIDATES = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Revenue",
    ],
    "Operating_Income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "Net_Income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "Diluted_EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "BasicAndDilutedEarningsPerShare",
    ],
    "Diluted_Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ],
    "OCF": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "Capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForProceedsFromOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}


def find_statement_value(statement_df, candidates):
    """
    Find a statement value using standardized XBRL concept/label information.

    This function deliberately does NOT take the first column containing a year.
    Instead, it examines the dataframe for period columns and chooses the most
    recent standalone-looking period available in the statement.
    """
    if statement_df is None or statement_df.empty:
        return np.nan

    df = statement_df.copy()

    # Locate a concept/label column.
    concept_col = None
    for c in ["standard_concept", "concept", "label", "name"]:
        if c in df.columns:
            concept_col = c
            break

    # Some EdgarTools statement DataFrames use the index for labels/concepts.
    if concept_col is None:
        working = df.copy()
        working["_row_text"] = working.index.astype(str)
    else:
        working = df.copy()
        working["_row_text"] = working[concept_col].astype(str)

    # Score rows by exact concept first, then normalized label.
    for candidate in candidates:
        candidate_norm = normalize_label(candidate)

        exact = working[
            working["_row_text"].map(normalize_label) == candidate_norm
        ]
        if not exact.empty:
            row = exact.iloc[0]
            value = _latest_numeric_period_value(row)
            if is_reasonable_numeric(value):
                return value

    # Conservative fallback: only use a label match if it is unambiguous.
    candidate_words = [normalize_label(x) for x in candidates]
    mask = working["_row_text"].map(
        lambda x: any(cw in normalize_label(x) for cw in candidate_words)
    )
    matches = working[mask]

    if len(matches) == 1:
        return _latest_numeric_period_value(matches.iloc[0])

    return np.nan


def _latest_numeric_period_value(row):
    """
    Select a numeric value from a statement row.

    Period columns are identified from their names and then ordered so that
    current/recent period columns are preferred. We avoid assuming that the
    first year-looking column is the desired quarter.
    """
    candidates = []

    for col in row.index:
        if str(col).startswith("_"):
            continue

        value = clean_number(row[col])
        if not is_reasonable_numeric(value):
            continue

        text = str(col)
        # Ignore obvious metadata columns.
        if any(x in text.lower() for x in ["label", "concept", "standard_concept", "units"]):
            continue

        # Score period-like columns. Current/recent columns get higher scores.
        score = 0
        upper = text.upper()

        if "Q" in upper:
            score += 10
        if "FY" in upper:
            score += 3
        if "202" in upper:
            score += 2

        # Prefer columns that appear toward the right/current end of the
        # statement when scores are otherwise similar.
        candidates.append((score, text, value))

    if not candidates:
        return np.nan

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _period_from_filing(filing):
    """Get filing report date safely."""
    try:
        return pd.to_datetime(str(filing.period_of_report)[:10], errors="coerce")
    except Exception:
        return pd.NaT


def _extract_from_edgar_statements(ticker):
    """
    Primary SEC path.

    Uses recent 10-Q/10-K/20-F filings and EdgarTools' parsed XBRL statements.
    It intentionally keeps only one observation per report date and uses the
    filing period date as the normalized quarter/year endpoint.
    """
    records = []
    diagnostics = []

    try:
        company = Company(ticker)

        # Get both domestic and foreign annual/quarterly reporting forms.
        # Foreign private issuers such as MELI/NU often use 20-F/6-K rather
        # than 10-Q, so the 10-Q-only approach is intentionally avoided.
        filings = company.get_filings(
            form=["10-Q", "10-K", "20-F"]
        )

        # Keep a reasonable number of recent filings. Sorting is handled below.
        filings = list(filings)[:60]

        for filing in filings:
            period = _period_from_filing(filing)
            if pd.isna(period):
                continue

            form = str(getattr(filing, "form", ""))

            try:
                obj = filing.obj()

                inc = getattr(obj, "income_statement", None)
                cf = getattr(obj, "cash_flow_statement", None)

                inc_df = (
                    inc.to_dataframe(view="standard")
                    if inc is not None
                    else None
                )
                cf_df = (
                    cf.to_dataframe(view="standard")
                    if cf is not None
                    else None
                )

                if inc_df is None or inc_df.empty:
                    continue

                record = {
                    "Period": period,
                    "Form": form,
                    "Source": "SEC XBRL",
                    "Revenue": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Revenue"]
                    ),
                    "Operating_Income": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Operating_Income"]
                    ),
                    "Net_Income": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Net_Income"]
                    ),
                    "Diluted_EPS": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Diluted_EPS"]
                    ),
                    "Diluted_Shares": find_statement_value(
                        inc_df, CONCEPT_CANDIDATES["Diluted_Shares"]
                    ),
                    "OCF": find_statement_value(
                        cf_df, CONCEPT_CANDIDATES["OCF"]
                    ),
                    "Capex": find_statement_value(
                        cf_df, CONCEPT_CANDIDATES["Capex"]
                    ),
                }

                # Require at least revenue or net income to count a filing.
                if is_reasonable_numeric(record["Revenue"]) or is_reasonable_numeric(
                    record["Net_Income"]
                ):
                    records.append(record)

            except Exception as exc:
                diagnostics.append(f"{period.date()} {form}: {type(exc).__name__}: {exc}")
                continue

    except Exception as exc:
        diagnostics.append(f"SEC company/fillings error: {type(exc).__name__}: {exc}")

    if not records:
        return pd.DataFrame(), diagnostics

    df = pd.DataFrame(records)

    # Multiple filings can contain the same report date. Prefer 10-Q for
    # quarterly dates, otherwise retain the first complete observation.
    df["completeness"] = df.notna().sum(axis=1)

    df = (
        df.sort_values(["Period", "completeness"], ascending=[True, False])
        .drop_duplicates("Period", keep="first")
        .drop(columns=["completeness"])
        .reset_index(drop=True)
    )

    return df, diagnostics


# ---------------------------------------------------------------------------
# YAHOO FALLBACK / SUPPLEMENT
# ---------------------------------------------------------------------------

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
        "Basic EPS",
    ],
    "Diluted_Shares": [
        "Diluted Average Shares",
        "Diluted Average Shares Outstanding",
        "Basic Average Shares",
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
                value = df.loc[key, date_col]
                return clean_number(value)
            except Exception:
                continue
    return np.nan


def fetch_yahoo_quarterly(ticker):
    """
    Yahoo fallback.

    Yahoo financial statements are already presented as quarterly columns, so
    we do not attempt the old SEC-style YTD subtraction here.
    """
    diagnostics = []

    try:
        tk = yf.Ticker(ticker)

        inc = tk.quarterly_income_stmt
        cf = tk.quarterly_cashflow

        if inc is None or inc.empty:
            return pd.DataFrame(), ["Yahoo quarterly income statement empty"]

        records = []

        for date_col in inc.columns:
            period = pd.to_datetime(date_col, errors="coerce")
            if pd.isna(period):
                continue

            records.append(
                {
                    "Period": period,
                    "Form": "Yahoo",
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

        return pd.DataFrame(records), diagnostics

    except Exception as exc:
        diagnostics.append(f"Yahoo error: {type(exc).__name__}: {exc}")
        return pd.DataFrame(), diagnostics


def merge_sec_and_yahoo(sec_df, yf_df):
    """
    SEC is authoritative where a value exists. Yahoo fills missing fields or
    missing periods. This is much safer than replacing the whole SEC dataset
    with Yahoo when one SEC field is missing.
    """
    if sec_df.empty:
        return yf_df.copy()

    if yf_df.empty:
        return sec_df.copy()

    all_cols = [
        "Period",
        "Form",
        "Source",
        "Revenue",
        "Operating_Income",
        "Net_Income",
        "Diluted_EPS",
        "Diluted_Shares",
        "OCF",
        "Capex",
    ]

    sec = sec_df.copy()
    yahoo = yf_df.copy()

    for df in [sec, yahoo]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan

    combined = pd.concat(
        [sec[all_cols], yahoo[all_cols]],
        ignore_index=True,
    )

    # Group by reporting date. SEC rows are placed first.
    combined["source_priority"] = np.where(
        combined["Source"].eq("SEC XBRL"), 0, 1
    )

    combined = combined.sort_values(
        ["Period", "source_priority"]
    )

    output = []

    for period, group in combined.groupby("Period", sort=True):
        row = {
            "Period": period,
            "Form": group.iloc[0]["Form"],
            "Source": group.iloc[0]["Source"],
        }

        for col in [
            "Revenue",
            "Operating_Income",
            "Net_Income",
            "Diluted_EPS",
            "Diluted_Shares",
            "OCF",
            "Capex",
        ]:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[col] = values.iloc[0] if not values.empty else np.nan

        # If Yahoo supplied a missing SEC value, identify the source as mixed.
        if len(group) > 1:
            sec_row = group[group["Source"].eq("SEC XBRL")]
            yf_row = group[group["Source"].eq("Yahoo Finance")]

            if not sec_row.empty and not yf_row.empty:
                for col in [
                    "Revenue",
                    "Operating_Income",
                    "Net_Income",
                    "Diluted_EPS",
                    "Diluted_Shares",
                    "OCF",
                    "Capex",
                ]:
                    sec_value = pd.to_numeric(
                        sec_row.iloc[0][col], errors="coerce"
                    )
                    yf_value = pd.to_numeric(
                        yf_row.iloc[0][col], errors="coerce"
                    )

                    if pd.isna(sec_value) and pd.notna(yf_value):
                        row[col] = yf_value
                        row["Source"] = "SEC + Yahoo"

        output.append(row)

    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# CASH FLOW NORMALIZATION
# ---------------------------------------------------------------------------

def calculate_fcf_safe(df):
    """
    Calculate standalone FCF.

    Important: Yahoo's quarterly cash flow values are already quarterly.
    SEC statement output may contain a mixture of quarterly and YTD periods.
    This function uses a conservative approach: when the extracted cash flow
    appears to be cumulative, subtract the prior same-fiscal-year observation.
    """
    if df.empty:
        return pd.DataFrame(columns=["Period", "FCF", "FCF_B"])

    work = df[["Period", "OCF", "Capex"]].copy()
    work["Period"] = pd.to_datetime(work["Period"], errors="coerce")
    work["OCF"] = pd.to_numeric(work["OCF"], errors="coerce")
    work["Capex"] = pd.to_numeric(work["Capex"], errors="coerce")
    work = work.sort_values("Period").reset_index(drop=True)

    # Because mixed SEC/Yahoo data may contain both quarterly and cumulative
    # values, only subtract when a current value is larger in magnitude than
    # the previous same-calendar-year value. This is intentionally conservative.
    standalone_ocf = []
    standalone_capex = []

    previous_year = None
    previous_ocf = np.nan
    previous_capex = np.nan

    for _, row in work.iterrows():
        period = row["Period"]
        year = period.year if pd.notna(period) else None

        ocf = row["OCF"]
        capex = row["Capex"]

        q_ocf = ocf
        q_capex = capex

        if year == previous_year:
            if pd.notna(ocf) and pd.notna(previous_ocf):
                # If the new value is clearly cumulative, derive the quarter.
                if abs(ocf) >= abs(previous_ocf) * 1.10:
                    q_ocf = ocf - previous_ocf

            if pd.notna(capex) and pd.notna(previous_capex):
                if abs(capex) >= abs(previous_capex) * 1.10:
                    q_capex = capex - previous_capex

        standalone_ocf.append(q_ocf)
        standalone_capex.append(q_capex)

        previous_year = year
        previous_ocf = ocf
        previous_capex = capex

    fcf = np.array(standalone_ocf, dtype=float) - np.abs(
        np.array(standalone_capex, dtype=float)
    )

    out = pd.DataFrame(
        {
            "Period": work["Period"],
            "FCF": fcf,
        }
    )

    out["FCF_B"] = out["FCF"] / 1e9
    out["FCF_YoY_%"] = pct_change_safe(out["FCF"], 4).clip(-200, 200)
    out["FCF_QoQ_%"] = pct_change_safe(out["FCF"], 1).clip(-200, 200)

    return out


# ---------------------------------------------------------------------------
# MAIN DATA FETCH
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_and_parse_ticker(ticker):
    sec_df, sec_diag = _extract_from_edgar_statements(ticker)
    yf_df, yf_diag = fetch_yahoo_quarterly(ticker)

    df = merge_sec_and_yahoo(sec_df, yf_df)

    if df.empty:
        raise ValueError(
            f"No usable quarterly financial data found for {ticker}. "
            f"SEC diagnostics: {sec_diag[-2:]}; Yahoo diagnostics: {yf_diag[-2:]}"
        )

    # Ensure all numeric fields are numeric.
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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("Period")
        .drop_duplicates("Period", keep="last")
        .reset_index(drop=True)
    )

    # The old code inferred shares from net income / EPS. Keep that only as a
    # last-resort fallback because it can be distorted by rounding or unusual
    # share-class accounting.
    implied_shares = safe_divide(df["Net_Income"], df["Diluted_EPS"])
    missing_shares = df["Diluted_Shares"].isna()
    df.loc[missing_shares, "Diluted_Shares"] = implied_shares[missing_shares]

    # Core derived metrics.
    df["Revenue_B"] = df["Revenue"] / 1e9
    df["Rev_YoY_%"] = pct_change_safe(df["Revenue_B"], 4)
    df["Rev_QoQ_%"] = pct_change_safe(df["Revenue_B"], 1)

    df["EPS_YoY_%"] = pct_change_safe(df["Diluted_EPS"], 4).clip(-200, 200)
    df["EPS_QoQ_%"] = pct_change_safe(df["Diluted_EPS"], 1).clip(-200, 200)

    df["Op_Margin_%"] = safe_divide(
        df["Operating_Income"], df["Revenue"]
    ) * 100

    df["Net_Margin_%"] = safe_divide(
        df["Net_Income"], df["Revenue"]
    ) * 100

    df["Capex_B"] = np.abs(df["Capex"]) / 1e9
    df["Diluted_Shares_M"] = df["Diluted_Shares"] / 1e6

    df["Share_Dilution_YoY_%"] = (
        pct_change_safe(df["Diluted_Shares_M"], 4).clip(-25, 25)
    )

    fcf_df = calculate_fcf_safe(df)

    return df, fcf_df, sec_diag, yf_diag


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_and_shares(ticker):
    try:
        tk = yf.Ticker(ticker)

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        market_cap = info.get("marketCap")
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        hist = tk.history(period="2y", auto_adjust=False)

        if not hist.empty and "Close" in hist.columns:
            hist = hist.copy()
            hist["EMA50"] = hist["Close"].ewm(
                span=50, adjust=False
            ).mean()
            hist["EMA200"] = hist["Close"].ewm(
                span=200, adjust=False
            ).mean()

            if not current_price:
                current_price = hist["Close"].iloc[-1]

        shares_outstanding = info.get("sharesOutstanding")

        if (
            not market_cap
            and current_price
            and shares_outstanding
        ):
            market_cap = current_price * shares_outstanding

        return hist, current_price, market_cap, shares_outstanding

    except Exception:
        return pd.DataFrame(), None, None, None


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

if run_button or ticker_symbol:
    try:
        with st.spinner(
            f"Extracting SEC/XBRL and market data for {ticker_symbol}..."
        ):
            (
                df_raw_full,
                df_fcf_full,
                sec_diag,
                yf_diag,
            ) = fetch_and_parse_ticker(ticker_symbol)

            (
                hist_price,
                current_price,
                market_cap,
                shares_outstanding,
            ) = fetch_market_and_shares(ticker_symbol)

        df_raw = df_raw_full.tail(lookback_quarters).reset_index(drop=True)
        df_fcf = df_fcf_full.tail(lookback_quarters).reset_index(drop=True)

        # -------------------------------------------------------------------
        # TTM / VALUATION
        # -------------------------------------------------------------------

        if market_cap and current_price:
            df_raw["TTM_Revenue"] = df_raw["Revenue"].rolling(4).sum()
            df_raw["Ann_Revenue"] = df_raw["Revenue"] * 4

            df_raw["P_S_TTM"] = (
                market_cap / df_raw["TTM_Revenue"]
            ).where(
                df_raw["TTM_Revenue"] > 0
            ).clip(lower=0, upper=150)

            df_raw["P_S_Ann"] = (
                market_cap / df_raw["Ann_Revenue"]
            ).where(
                df_raw["Ann_Revenue"] > 0
            ).clip(lower=0, upper=150)

            df_raw["TTM_EPS"] = df_raw["Diluted_EPS"].rolling(4).sum()
            df_raw["Ann_EPS"] = df_raw["Diluted_EPS"] * 4

            df_raw["P_E_TTM"] = (
                current_price / df_raw["TTM_EPS"]
            ).where(
                df_raw["TTM_EPS"] > 0
            ).clip(lower=0, upper=200)

            df_raw["P_E_Ann"] = (
                current_price / df_raw["Ann_EPS"]
            ).where(
                df_raw["Ann_EPS"] > 0
            ).clip(lower=0, upper=200)

            merged_fcf = df_raw[["Period"]].merge(
                df_fcf[["Period", "FCF"]],
                on="Period",
                how="left",
            )

            df_raw["TTM_FCF"] = merged_fcf["FCF"].rolling(4).sum()

            df_raw["FCF_Yield_%"] = (
                df_raw["TTM_FCF"] / market_cap
            ) * 100

        # -------------------------------------------------------------------
        # HEADER / DIAGNOSTICS
        # -------------------------------------------------------------------

        st.subheader(
            f"{ticker_symbol} — Executive Financial Dashboard"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Current Price",
                f"${current_price:,.2f}"
                if current_price
                else "N/A",
            )

        with c2:
            st.metric(
                "Market Cap",
                f"${market_cap/1e9:,.2f}B"
                if market_cap
                else "N/A",
            )

        with c3:
            st.metric(
                "Quarterly Records",
                f"{len(df_raw_full)}",
            )

        with c4:
            source_counts = df_raw_full["Source"].value_counts()
            st.metric(
                "SEC Records",
                f"{source_counts.get('SEC XBRL', 0) + source_counts.get('SEC + Yahoo', 0)}",
            )

        with st.expander("Data Source Diagnostics"):
            st.write("**Sources used:**")
            st.dataframe(
                df_raw_full[
                    ["Period", "Form", "Source"]
                ].tail(20),
                use_container_width=True,
            )

            if sec_diag:
                st.write("**SEC diagnostics (most recent):**")
                for msg in sec_diag[-10:]:
                    st.caption(msg)

            if yf_diag:
                st.write("**Yahoo diagnostics:**")
                for msg in yf_diag[-10:]:
                    st.caption(msg)

        # -------------------------------------------------------------------
        # FINANCIAL DASHBOARD
        # -------------------------------------------------------------------

        fig, axes = plt.subplots(
            2, 2, figsize=(16, 11), dpi=150
        )

        fig.suptitle(
            f"{ticker_symbol} Financial & Growth Dashboard",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Revenue
        ax1 = axes[0, 0]
        v_rev = df_raw.dropna(subset=["Revenue_B"])

        if not v_rev.empty:
            ax1.bar(
                v_rev["Period"].astype(str),
                v_rev["Revenue_B"],
                alpha=0.85,
                width=0.55,
                label="Revenue ($B)",
            )

            ax1.set_title(
                "Revenue ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax1.set_ylabel("Revenue ($ Billions)")
            ax1.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax1_sub = ax1.twinx()

            ax1_sub.plot(
                v_rev["Period"].astype(str),
                v_rev["Rev_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax1_sub.plot(
                v_rev["Period"].astype(str),
                v_rev["Rev_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax1_sub.set_ylabel("Growth (%)")
            ax1_sub.grid(False)

            l1, lb1 = ax1.get_legend_handles_labels()
            l1s, lb1s = ax1_sub.get_legend_handles_labels()

            ax1.legend(
                l1 + l1s,
                lb1 + lb1s,
                loc="upper left",
                fontsize=6.5,
            )

        # EPS
        ax2 = axes[0, 1]
        v_eps = df_raw.dropna(subset=["Diluted_EPS"])

        if not v_eps.empty:
            ax2.bar(
                v_eps["Period"].astype(str),
                v_eps["Diluted_EPS"],
                alpha=0.85,
                width=0.55,
                label="Diluted EPS ($)",
            )

            ax2.set_title(
                "Diluted EPS ($) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax2.set_ylabel("EPS ($)")
            ax2.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax2_sub = ax2.twinx()

            ax2_sub.plot(
                v_eps["Period"].astype(str),
                v_eps["EPS_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax2_sub.plot(
                v_eps["Period"].astype(str),
                v_eps["EPS_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax2_sub.set_ylabel("Growth (%)")
            ax2_sub.grid(False)

            l2, lb2 = ax2.get_legend_handles_labels()
            l2s, lb2s = ax2_sub.get_legend_handles_labels()

            ax2.legend(
                l2 + l2s,
                lb2 + lb2s,
                loc="upper left",
                fontsize=6.5,
            )

        # FCF
        ax3 = axes[1, 0]
        v_fcf = df_fcf.tail(lookback_quarters).copy()
        v_fcf_clean = v_fcf.dropna(subset=["FCF_B"])

        if not v_fcf_clean.empty:
            ax3.bar(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_B"],
                alpha=0.85,
                width=0.55,
                label="Free Cash Flow ($B)",
            )

            ax3.set_title(
                "Standalone Free Cash Flow ($B) & Growth",
                fontweight="bold",
                fontsize=10.5,
            )
            ax3.set_ylabel("FCF ($ Billions)")
            ax3.tick_params(
                axis="x",
                rotation=45,
                labelsize=7,
            )

            ax3_sub = ax3.twinx()

            ax3_sub.plot(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_YoY_%"],
                marker="o",
                linewidth=1.5,
                label="YoY Growth (%)",
            )

            ax3_sub.plot(
                v_fcf_clean["Period"].astype(str),
                v_fcf_clean["FCF_QoQ_%"],
                marker="s",
                linestyle="--",
                linewidth=1.2,
                label="QoQ Growth (%)",
            )

            ax3_sub.set_ylabel("Growth (%)")
            ax3_sub.grid(False)

            l3, lb3 = ax3.get_legend_handles_labels()
            l3s, lb3s = ax3_sub.get_legend_handles_labels()

            ax3.legend(
                l3 + l3s,
                lb3 + lb3s,
                loc="upper left",
                fontsize=6.5,
            )

        # Margins
        ax4 = axes[1, 1]

        ax4.plot(
            df_raw["Period"].astype(str),
            df_raw["Op_Margin_%"],
            marker="o",
            linewidth=2,
            label="Operating Margin (%)",
        )

        ax4.plot(
            df_raw["Period"].astype(str),
            df_raw["Net_Margin_%"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Net Margin (%)",
        )

        ax4.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax4.set_title(
            "Operating Margin vs. Net Margin (%)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax4.set_ylabel("Margin (%)")
        ax4.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax4.legend(
            loc="upper left",
            fontsize=7,
        )
        ax4.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )
        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------------------------
        # VALUATION
        # -------------------------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Valuation Multiples & Price Action"
        )

        fig2, axes2 = plt.subplots(
            2, 2, figsize=(16, 11), dpi=150
        )

        fig2.suptitle(
            f"{ticker_symbol} Price Action, TTM/Annualized Multiples & FCF Yield",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Price / EMA
        ax_p1 = axes2[0, 0]

        if hist_price is not None and not hist_price.empty:
            ax_p1.plot(
                hist_price.index,
                hist_price["Close"],
                linewidth=1.5,
                label="Close Price ($)",
            )

            if "EMA50" in hist_price.columns:
                ax_p1.plot(
                    hist_price.index,
                    hist_price["EMA50"],
                    linestyle="-",
                    linewidth=1.2,
                    label="50-Day EMA",
                )

            if "EMA200" in hist_price.columns:
                ax_p1.plot(
                    hist_price.index,
                    hist_price["EMA200"],
                    linestyle="--",
                    linewidth=1.2,
                    label="200-Day EMA",
                )

        ax_p1.set_title(
            "Daily Stock Price vs 50/200 EMA",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p1.set_ylabel("Price ($)")
        ax_p1.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p1.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p1.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/S
        ax_p2 = axes2[0, 1]

        if "P_S_TTM" in df_raw.columns:
            ax_p2.plot(
                df_raw["Period"].astype(str),
                df_raw["P_S_TTM"],
                marker="o",
                linewidth=2,
                label="P/S (TTM)",
            )

            ax_p2.plot(
                df_raw["Period"].astype(str),
                df_raw["P_S_Ann"],
                marker="^",
                linestyle="--",
                linewidth=1.5,
                label="P/S (Annualized Quarter)",
            )

        ax_p2.set_title(
            "Price-to-Sales (P/S): TTM vs Annualized Q",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p2.set_ylabel("P/S Multiple (x)")
        ax_p2.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p2.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p2.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # P/E
        ax_p3 = axes2[1, 0]

        if "P_E_TTM" in df_raw.columns:
            ax_p3.plot(
                df_raw["Period"].astype(str),
                df_raw["P_E_TTM"],
                marker="s",
                linewidth=2,
                label="P/E (TTM)",
            )

            ax_p3.plot(
                df_raw["Period"].astype(str),
                df_raw["P_E_Ann"],
                marker="d",
                linestyle="--",
                linewidth=1.5,
                label="P/E (Annualized Quarter)",
            )

        ax_p3.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_p3.set_title(
            "Price-to-Earnings (P/E): TTM vs Annualized Q",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p3.set_ylabel("P/E Multiple (x)")
        ax_p3.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p3.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p3.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # FCF Yield
        ax_p4 = axes2[1, 1]

        if "FCF_Yield_%" in df_raw.columns:
            ax_p4.plot(
                df_raw["Period"].astype(str),
                df_raw["FCF_Yield_%"],
                marker="^",
                linewidth=2,
                label="FCF Yield (TTM %)",
            )

        ax_p4.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_p4.set_title(
            "Free Cash Flow Yield (TTM %)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_p4.set_ylabel("FCF Yield (%)")
        ax_p4.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_p4.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_p4.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )
        st.pyplot(fig2)
        plt.close(fig2)

        # -------------------------------------------------------------------
        # CAPEX / DILUTION
        # -------------------------------------------------------------------

        st.markdown("---")
        st.subheader(
            f"{ticker_symbol} — Capital Expenditures & Share Dilution"
        )

        fig3, axes3 = plt.subplots(
            1, 2, figsize=(16, 5), dpi=150
        )

        ax_c1 = axes3[0]

        ax_c1.bar(
            df_raw["Period"].astype(str),
            df_raw["Capex_B"],
            alpha=0.85,
            width=0.55,
            label="Capex ($B)",
        )

        ax_c1.set_title(
            "Quarterly Capital Expenditures ($B)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_c1.set_ylabel("Capex ($ Billions)")
        ax_c1.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_c1.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_c1.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        ax_c2 = axes3[1]

        ax_c2.plot(
            df_raw["Period"].astype(str),
            df_raw["Share_Dilution_YoY_%"],
            marker="o",
            linewidth=2,
            label="Diluted Share Growth YoY (%)",
        )

        ax_c2.axhline(
            0,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        ax_c2.set_title(
            "Share Dilution / Buyback Rate (YoY % Change)",
            fontweight="bold",
            fontsize=10.5,
        )
        ax_c2.set_ylabel("Share Count YoY Change (%)")
        ax_c2.tick_params(
            axis="x",
            rotation=45,
            labelsize=7,
        )
        ax_c2.legend(
            loc="upper left",
            fontsize=7,
        )
        ax_c2.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # -------------------------------------------------------------------
        # RAW DATA
        # -------------------------------------------------------------------

        with st.expander(
            "View Raw Extracted Dataset & Valuations"
        ):
            display_df = df_raw.copy()
            display_df["Period"] = display_df["Period"].dt.strftime("%Y-%m-%d")

            st.dataframe(
                display_df,
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            f"Could not load data for ticker '{ticker_symbol}'. "
            f"Error: {type(exc).__name__}: {exc}"
        )

        st.exception(exc)

      if curr_year != prev_year or month <= 4:
        q_ocf = ocf_val
        q_capex = capex_val
      else:
        q_ocf = ocf_val - prev_ocf_ytd
        q_capex = capex_val - prev_capex_ytd

      standalone_ocf.append(q_ocf)
      standalone_capex.append(q_capex)
      prev_year = curr_year
      prev_ocf_ytd = ocf_val
      prev_capex_ytd = capex_val

    fcf_arr = np.array(standalone_ocf) - np.abs(np.array(standalone_capex))
    df_fcf = pd.DataFrame({"Period": df_cf_raw["Period"], "FCF": fcf_arr})
    df_fcf["FCF_B"] = df_fcf["FCF"] / 1e9
    df_fcf["FCF_YoY_%"] = (
        df_fcf["FCF"].pct_change(periods=4, fill_method=None).clip(-200, 200)
        * 100
    )
    df_fcf["FCF_QoQ_%"] = (
        df_fcf["FCF"].pct_change(periods=1, fill_method=None).clip(-200, 200)
        * 100
    )
    return df_fcf
  except Exception:
    return pd.DataFrame({"Period": df["Period"], "FCF_B": np.nan})


@st.cache_data(ttl=86400)
def fetch_and_parse_ticker(ticker):
  records = []
  edgar_success = False

  try:
    company = Company(ticker)
    filings = company.get_filings(form="10-Q")
    if len(filings) < 4:
      filings = company.get_filings()[:40]
    else:
      filings = filings[:40]

    def parse_multi_val(df, keywords, min_val=None):
      if df is None:
        return np.nan
      for kw in keywords:
        match = df[df["label"].str.contains(kw, case=False, na=False)]
        if not match.empty:
          num_cols = [c for c in df.columns if "202" in str(c) or "201" in str(c)]
          if num_cols:
            val_raw = match.iloc[0][num_cols[0]]
            if pd.notna(val_raw):
              val_str = (
                  str(val_raw)
                  .replace("$", "")
                  .replace(",", "")
                  .replace("(", "-")
                  .replace(")", "")
              )
              try:
                val = float(val_str)
                if min_val is not None and val < min_val:
                  continue
                return val
              except:
                pass
      return np.nan

    for f in filings:
      try:
        obj = f.obj()
        inc = obj.income_statement
        cf = obj.cash_flow_statement
        if inc is None:
          continue
        inc_df = inc.to_dataframe(view="standard")
        cf_df = cf.to_dataframe(view="standard") if cf is not None else None

        rev = parse_multi_val(
            inc_df, [
                "Revenue",
                "Total revenue",
                "Revenues, net",
                "Net revenues",
                "Total net revenues",
            ]
        )
        op_inc = parse_multi_val(
            inc_df,
            [
                "Income from operations",
                "Loss from operations",
                "Income (loss) from operations",
                "Operating income (loss)",
                "Profit from operations",
                "Operating profit",
            ],
        )
        net_inc = parse_multi_val(
            inc_df,
            [
                "Net income including",
                "Net income (loss)",
                "Net loss",
                "Net income",
                "Profit (loss) for the period",
            ],
        )
        eps_diluted = parse_multi_val(
            inc_df,
            [
                "Diluted earnings per share",
                "Earnings per share, diluted",
                "Diluted (in USD per share)",
                "Basic and diluted",
                "Diluted",
                "Earnings per share - diluted",
                "Diluted earnings (loss) per share",
            ],
        )
        diluted_shares = parse_multi_val(
            inc_df,
            [
                "Weighted-average shares outstanding, diluted",
                "Weighted average shares outstanding, diluted",
                "Weighted average number of shares outstanding, diluted",
                "Weighted average shares diluted",
                "Diluted shares",
                "Number of diluted shares",
            ],
            min_val=100000,
        )

        ocf = parse_multi_val(
            cf_df, [
                "Net cash provided by operating activities",
                "Net cash provided by (used in) operating activities",
                "Operating cash flow",
            ]
        )
        capex = parse_multi_val(
            cf_df, [
                "Purchases of property and equipment",
                "Additions to property and equipment",
                "Capital expenditures",
            ]
        )

        records.append({
            "Period": str(f.period_of_report)[:10],
            "Revenue": rev,
            "Operating_Income": op_inc,
            "Net_Income": net_inc,
            "Diluted_EPS": eps_diluted,
            "Diluted_Shares": diluted_shares,
            "OCF": ocf,
            "Capex": capex,
        })
      except Exception:
        continue
    if len(records) >= 4:
      edgar_success = True
  except Exception:
    edgar_success = False

  if not edgar_success or len(records) < 4:
    records = []
    try:
      tk = yf.Ticker(ticker)
      q_inc = tk.quarterly_income_stmt
      q_cf = tk.quarterly_cashflow
      q_bal = tk.quarterly_balance_sheet

      if q_inc is not None and not q_inc.empty:
        for date_col in q_inc.columns:
          p_str = str(date_col)[:10]

          def get_val(df_q, keys):
            if df_q is None or df_q.empty:
              return np.nan
            for k in keys:
              if k in df_q.index:
                v = df_q.loc[k, date_col]
                if pd.notna(v):
                  return float(v)
            return np.nan

          rev = get_val(
              q_inc, [
                  "Total Revenue",
                  "Total Revenues",
                  "Operating Revenue",
                  "Revenue",
              ]
          )
          op_inc = get_val(
              q_inc, ["Operating Income", "EBIT", "Operating Profit"]
          )
          net_inc = get_val(
              q_inc, ["Net Income", "Net Income Common Stockholders"]
          )
          eps_dil = get_val(
              q_inc, [
                  "Diluted EPS",
                  "Basic EPS",
                  "Diluted Earnings Per Share",
              ]
          )
          shares = get_val(
              q_bal, [
                  "Ordinary Shares Number",
                  "Share Issued",
                  "Common Stock",
                  "Diluted Average Shares",
              ]
          )
          ocf = get_val(
              q_cf, [
                  "Operating Cash Flow",
                  "Cash Flow From Continuing Operating Activities",
              ]
          )
          capex = get_val(
              q_cf, ["Capital Expenditure", "Purchase Of Property And Equipment"]
          )

          records.append({
              "Period": p_str,
              "Revenue": rev,
              "Operating_Income": op_inc,
              "Net_Income": net_inc,
              "Diluted_EPS": eps_dil,
              "Diluted_Shares": shares,
              "OCF": ocf,
              "Capex": capex,
          })
    except Exception:
      pass

  df = (
      pd.DataFrame(records)
      .sort_values("Period")
      .drop_duplicates(subset=["Period"])
      .reset_index(drop=True)
  )

  if df.empty:
    raise ValueError(
        f"No valid financial periods could be parsed for {ticker}."
    )

  if "Diluted_EPS" in df.columns:
    df["Diluted_EPS"] = pd.to_numeric(df["Diluted_EPS"], errors="coerce")
  if "Diluted_Shares" in df.columns:
    df["Diluted_Shares"] = pd.to_numeric(df["Diluted_Shares"], errors="coerce")

  implied_shares = df["Net_Income"] / df["Diluted_EPS"]
  df["Diluted_Shares"] = df["Diluted_Shares"].fillna(implied_shares)

  df["Revenue_B"] = df["Revenue"] / 1e9
  df["Rev_YoY_%"] = df["Revenue_B"].pct_change(periods=4, fill_method=None) * 100
  df["Rev_QoQ_%"] = df["Revenue_B"].pct_change(periods=1, fill_method=None) * 100

  eps_yoy_raw = df["Diluted_EPS"].pct_change(periods=4, fill_method=None) * 100
  eps_qoq_raw = df["Diluted_EPS"].pct_change(periods=1, fill_method=None) * 100
  df["EPS_YoY_%"] = eps_yoy_raw.clip(-200, 200)
  df["EPS_QoQ_%"] = eps_qoq_raw.clip(-200, 200)

  df["Op_Margin_%"] = (df["Operating_Income"] / df["Revenue"]) * 100
  df["Net_Margin_%"] = (df["Net_Income"] / df["Revenue"]) * 100

  df["Capex_B"] = np.abs(df["Capex"]) / 1e9
  df["Diluted_Shares_M"] = (
      pd.to_numeric(df["Diluted_Shares"], errors="coerce") / 1e6
  )
  df["Share_Dilution_YoY_%"] = (
      df["Diluted_Shares_M"].pct_change(periods=4, fill_method=None).clip(-25, 25)
      * 100
  )

  df_fcf = calculate_fcf_safe(df)
  return df, df_fcf


@st.cache_data(ttl=3600)
def fetch_market_and_shares(ticker):
  try:
    tk = yf.Ticker(ticker)
    info = tk.info
    market_cap = info.get("marketCap")
    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    hist = tk.history(period="2y")
    if not hist.empty:
      hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
      hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()
      if not current_price:
        current_price = hist["Close"].iloc[-1]
    if not market_cap and current_price and info.get("sharesOutstanding"):
      market_cap = current_price * info.get("sharesOutstanding")
    return hist, current_price, market_cap
  except Exception:
    return pd.DataFrame(), None, None


try:
  with st.spinner(
      f"Extracting financial statements & market data for {ticker_symbol}..."
  ):
    df_raw_full, df_fcf_full = fetch_and_parse_ticker(ticker_symbol)
    hist_price, current_price, market_cap = fetch_market_and_shares(
        ticker_symbol
    )

  df_raw = df_raw_full.tail(lookback_quarters).reset_index(drop=True)
  df_fcf = df_fcf_full.tail(lookback_quarters).reset_index(drop=True)

  if market_cap and current_price:
    df_raw["TTM_Revenue"] = df_raw["Revenue"].rolling(4).sum()
    df_raw["Ann_Revenue"] = df_raw["Revenue"] * 4
    df_raw["P_S_TTM"] = (market_cap / df_raw["TTM_Revenue"]).clip(
        lower=0, upper=150
    )
    df_raw["P_S_Ann"] = (market_cap / df_raw["Ann_Revenue"]).clip(
        lower=0, upper=150
    )

    df_raw["TTM_EPS"] = df_raw["Diluted_EPS"].rolling(4).sum()
    df_raw["Ann_EPS"] = df_raw["Diluted_EPS"] * 4

    df_raw["P_E_TTM"] = (current_price / df_raw["TTM_EPS"]).clip(-150, 200)
    df_raw["P_E_Ann"] = (current_price / df_raw["Ann_EPS"]).clip(-150, 200)

    merged_fcf = df_raw[["Period"]].merge(
        df_fcf[["Period", "FCF"]], on="Period", how="left"
    )
    df_raw["TTM_FCF"] = merged_fcf["FCF"].rolling(4).sum()
    df_raw["FCF_Yield_%"] = (df_raw["TTM_FCF"] / market_cap) * 100

  st.subheader(f"{ticker_symbol} — Executive 2x2 Financial Dashboard")

  fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
  fig.suptitle(
      f"{ticker_symbol} Comprehensive Financial & Growth Dashboard",
      fontsize=15,
      fontweight="bold",
      y=0.98,
  )

  # 1. Revenue
  ax1 = axes[0, 0]
  v_rev = df_raw.dropna(subset=["Revenue_B"])
  if not v_rev.empty:
    ax1.bar(
        v_rev["Period"],
        v_rev["Revenue_B"],
        color="#8B5CF6",
        alpha=0.85,
        width=0.55,
        label="Revenue ($B)",
    )
    ax1.set_title(
        "Revenue ($B) & YoY/QoQ Growth", fontweight="bold", fontsize=10.5
    )
    ax1.set_ylabel("Revenue ($ Billions)", color="#6D28D9")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1_sub = ax1.twinx()
    ax1_sub.plot(
        v_rev["Period"],
        v_rev["Rev_YoY_%"],
        color="#EF4444",
        marker="o",
        linewidth=1.5,
        label="YoY Growth (%)",
    )
    ax1_sub.plot(
        v_rev["Period"],
        v_rev["Rev_QoQ_%"],
        color="#10B981",
        marker="s",
        linestyle="--",
        linewidth=1.2,
        label="QoQ Growth (%)",
    )
    ax1_sub.set_ylabel("Growth (%)", fontsize=8)
    ax1_sub.grid(False)
    l1, lb1 = ax1.get_legend_handles_labels()
    l1s, lb1s = ax1_sub.get_legend_handles_labels()
    ax1.legend(l1 + l1s, lb1 + lb1s, loc="upper left", fontsize=6.5)

  # 2. EPS
  ax2 = axes[0, 1]
  v_eps = df_raw.dropna(subset=["Diluted_EPS"])
  if not v_eps.empty:
    ax2.bar(
        v_eps["Period"],
        v_eps["Diluted_EPS"],
        color="#3B82F6",
        alpha=0.85,
        width=0.55,
        label="Diluted EPS ($)",
    )
    ax2.set_title("Diluted EPS ($) & Growth", fontweight="bold", fontsize=10.5)
    ax2.set_ylabel("EPS ($)", color="#1D4ED8")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2_sub = ax2.twinx()
    ax2_sub.plot(
        v_eps["Period"],
        v_eps["EPS_YoY_%"],
        color="#EF4444",
        marker="o",
        linewidth=1.5,
        label="YoY Growth (%)",
    )
    ax2_sub.plot(
        v_eps["Period"],
        v_eps["EPS_QoQ_%"],
        color="#10B981",
        marker="s",
        linestyle="--",
        linewidth=1.2,
        label="QoQ Growth (%)",
    )
    ax2_sub.set_ylabel("Growth (%)", fontsize=8)
    ax2_sub.grid(False)
    l2, lb2 = ax2.get_legend_handles_labels()
    l2s, lb2s = ax2_sub.get_legend_handles_labels()
    ax2.legend(l2 + l2s, lb2 + lb2s, loc="upper left", fontsize=6.5)

  # 3. FCF
  ax3 = axes[1, 0]
  v_fcf = df_fcf.tail(lookback_quarters).reset_index(drop=True)
  v_fcf_clean = v_fcf.dropna(subset=["FCF_B"])
  if not v_fcf_clean.empty:
    ax3.bar(
        v_fcf_clean["Period"],
        v_fcf_clean["FCF_B"],
        color="#10B981",
        alpha=0.85,
        width=0.55,
        label="Free Cash Flow ($B)",
    )
    ax3.set_title(
        "Standalone Free Cash Flow ($B) & Growth",
        fontweight="bold",
        fontsize=10.5,
    )
    ax3.set_ylabel("FCF ($ Billions)", color="#047857")
    ax3.tick_params(axis="x", rotation=45, labelsize=7)
    ax3_sub = ax3.twinx()
    ax3_sub.plot(
        v_fcf_clean["Period"],
        v_fcf_clean["FCF_YoY_%"],
        color="#EF4444",
        marker="o",
        linewidth=1.5,
        label="YoY Growth (%)",
    )
    ax3_sub.plot(
        v_fcf_clean["Period"],
        v_fcf_clean["FCF_QoQ_%"],
        color="#3B82F6",
        marker="s",
        linestyle="--",
        linewidth=1.2,
        label="QoQ Growth (%)",
    )
    ax3_sub.set_ylabel("Growth (%)", fontsize=8)
    ax3_sub.grid(False)
    l3, lb3 = ax3.get_legend_handles_labels()
    l3s, lb3s = ax3_sub.get_legend_handles_labels()
    ax3.legend(l3 + l3s, lb3 + lb3s, loc="upper left", fontsize=6.5)

  # 4. Margins
  ax4 = axes[1, 1]
  ax4.plot(
      df_raw["Period"],
      df_raw["Op_Margin_%"],
      color="#3B82F6",
      marker="o",
      linewidth=2,
      label="Operating Margin (%)",
  )
  ax4.plot(
      df_raw["Period"],
      df_raw["Net_Margin_%"],
      color="#8B5CF6",
      marker="s",
      linestyle="--",
      linewidth=2,
      label="Net Margin (%)",
  )
  ax4.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
  ax4.set_title(
      "Operating Margin vs. Net Margin (%)", fontweight="bold", fontsize=10.5
  )
  ax4.set_ylabel("Margin (%)", color="#1F2937")
  ax4.tick_params(axis="x", rotation=45, labelsize=7)
  ax4.legend(loc="upper left", fontsize=7)
  ax4.grid(True, linestyle="--", alpha=0.3)

  plt.tight_layout(rect=[0, 0, 1, 0.98])
  st.pyplot(fig)

  st.markdown("---")
  st.subheader(f"{ticker_symbol} — Valuation Multiples & Price Action")

  fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
  fig2.suptitle(
      f"{ticker_symbol} Price Action, TTM/Annualized Multiples & FCF Yield",
      fontsize=15,
      fontweight="bold",
      y=0.98,
  )

  # Valuation Chart 1: Stock Price + 50/200 EMA
  ax_p1 = axes2[0, 0]
  if hist_price is not None and not hist_price.empty:
    ax_p1.plot(
        hist_price.index,
        hist_price["Close"],
        color="#1F2937",
        linewidth=1.5,
        label="Close Price ($)",
    )
    if "EMA50" in hist_price.columns:
      ax_p1.plot(
          hist_price.index,
          hist_price["EMA50"],
          color="#3B82F6",
          linestyle="-",
          linewidth=1.2,
          label="50-Day EMA",
      )
    if "EMA200" in hist_price.columns:
      ax_p1.plot(
          hist_price.index,
          hist_price["EMA200"],
          color="#EF4444",
          linestyle="--",
          linewidth=1.2,
          label="200-Day EMA",
      )
  ax_p1.set_title("Daily Stock Price vs 50/200 EMA", fontweight="bold", fontsize=10.5)
  ax_p1.set_ylabel("Price ($)", color="#1F2937")
  ax_p1.tick_params(axis="x", rotation=45, labelsize=7)
  ax_p1.legend(loc="upper left", fontsize=7)
  ax_p1.grid(True, linestyle="--", alpha=0.3)

  # Valuation Chart 2: P/S Ratio
  ax_p2 = axes2[0, 1]
  ax_p2.plot(
      df_raw["Period"],
      df_raw["P_S_TTM"],
      color="#8B5CF6",
      marker="o",
      linewidth=2,
      label="P/S (TTM)",
  )
  ax_p2.plot(
      df_raw["Period"],
      df_raw["P_S_Ann"],
      color="#A78BFA",
      marker="^",
      linestyle="--",
      linewidth=1.5,
      label="P/S (Annualized Quarter)",
  )
  ax_p2.set_title(
      "Price-to-Sales (P/S): TTM vs. Annualized Q", fontweight="bold", fontsize=10.5
  )
  ax_p2.set_ylabel("P/S Multiple (x)", color="#6D28D9")
  ax_p2.tick_params(axis="x", rotation=45, labelsize=7)
  ax_p2.legend(loc="upper left", fontsize=7)
  ax_p2.grid(True, linestyle="--", alpha=0.3)

  # Valuation Chart 3: P/E Ratio
  ax_p3 = axes2[1, 0]
  ax_p3.plot(
      df_raw["Period"],
      df_raw["P_E_TTM"],
      color="#10B981",
      marker="s",
      linewidth=2,
      label="P/E (TTM)",
  )
  ax_p3.plot(
      df_raw["Period"],
      df_raw["P_E_Ann"],
      color="#34D399",
      marker="d",
      linestyle="--",
      linewidth=1.5,
      label="P/E (Annualized Quarter)",
  )
  ax_p3.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
  ax_p3.set_title(
      "Price-to-Earnings (P/E): TTM vs. Annualized Q",
      fontweight="bold",
      fontsize=10.5,
  )
  ax_p3.set_ylabel("P/E Multiple (x)", color="#047857")
  ax_p3.tick_params(axis="x", rotation=45, labelsize=7)
  ax_p3.legend(loc="upper left", fontsize=7)
  ax_p3.grid(True, linestyle="--", alpha=0.3)

  # Valuation Chart 4: Dedicated Free Cash Flow Yield (%)
  ax_p4 = axes2[1, 1]
  if "FCF_Yield_%" in df_raw.columns:
    ax_p4.plot(
        df_raw["Period"],
        df_raw["FCF_Yield_%"],
        color="#F59E0B",
        marker="^",
        linewidth=2,
        label="FCF Yield (TTM %)",
    )
  ax_p4.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
  ax_p4.set_title("Free Cash Flow Yield (TTM %)", fontweight="bold", fontsize=10.5)
  ax_p4.set_ylabel("FCF Yield (%)", color="#B45309")
  ax_p4.tick_params(axis="x", rotation=45, labelsize=7)
  ax_p4.legend(loc="upper left", fontsize=7)
  ax_p4.grid(True, linestyle="--", alpha=0.3)

  plt.tight_layout(rect=[0, 0, 1, 0.98])
  st.pyplot(fig2)

  # Bottom Section: Capital Structure & Dilution Rate vs Capex
  st.markdown("---")
  st.subheader(f"{ticker_symbol} — Capital Expenditures & Share Dilution Rate")

  fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5), dpi=150)

  # Panel A: Capex ($B)
  ax_c1 = axes3[0]
  ax_c1.bar(
      df_raw["Period"],
      df_raw["Capex_B"],
      color="#F59E0B",
      alpha=0.85,
      width=0.55,
      label="Capex ($B)",
  )
  ax_c1.set_title(
      "Quarterly Capital Expenditures ($B)", fontweight="bold", fontsize=10.5
  )
  ax_c1.set_ylabel("Capex ($ Billions)", color="#B45309")
  ax_c1.tick_params(axis="x", rotation=45, labelsize=7)
  ax_c1.legend(loc="upper left", fontsize=7)
  ax_c1.grid(True, linestyle="--", alpha=0.3)

  # Panel B: Share Dilution Rate YoY (%)
  ax_c2 = axes3[1]
  ax_c2.plot(
      df_raw["Period"],
      df_raw["Share_Dilution_YoY_%"],
      color="#2563EB",
      marker="o",
      linewidth=2,
      label="Diluted Share Growth YoY (%)",
  )
  ax_c2.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
  ax_c2.set_title(
      "Share Dilution / Buyback Rate (YoY % Change)",
      fontweight="bold",
      fontsize=10.5,
  )
  ax_c2.set_ylabel("Share Count YoY Change (%)", color="#1D4ED8")
  ax_c2.tick_params(axis="x", rotation=45, labelsize=7)
  ax_c2.legend(loc="upper left", fontsize=7)
  ax_c2.grid(True, linestyle="--", alpha=0.3)

  plt.tight_layout()
  st.pyplot(fig3)

  with st.expander("View Raw Extracted Dataset & Valuations"):
    st.dataframe(df_raw, use_container_width=True)

except Exception as e:
  st.error(
      f"Could not load data for ticker '{ticker_symbol}'. Check ticker symbol or"
      f" network connection. Error: {e}"
  )
