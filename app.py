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

    # Convert pandas NA/None/strings safely before NumPy arithmetic.
    # np.array([...], dtype=float) raises TypeError when a list contains pd.NA.
    ocf_series = pd.to_numeric(
        pd.Series(standalone_ocf, index=work.index),
        errors="coerce",
    ).astype(float)

    capex_series = pd.to_numeric(
        pd.Series(standalone_capex, index=work.index),
        errors="coerce",
    ).astype(float)

    fcf = ocf_series.to_numpy() - np.abs(capex_series.to_numpy())

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
            f"Could not load data for ticker '{ticker_symbol}'. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        st.exception(exc)
