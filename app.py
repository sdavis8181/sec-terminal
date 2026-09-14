from pathlib import Path
from edgar import Company, set_identity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

set_identity("Scott Scott@example.com")

st.set_page_config(
    page_title="SEC XBRL Financial Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Institutional SEC XBRL Financial Dashboard")
st.markdown(
    "Enter a stock ticker to pull live GAAP financial statements, disaggregate"
    " standalone quarterly cash flows, and generate executive growth charts."
)

with st.sidebar:
  st.header("Terminal Controls")
  ticker_symbol = (
      st.text_input("Stock Ticker", value="UBER", max_chars=10)
      .strip()
      .upper()
  )
  lookback_quarters = st.slider(
      "Historical Lookback (Quarters)", min_value=12, max_value=40, value=20, step=4
  )
  run_button = st.button("Generate Report", type="primary")

  st.markdown("---")
  st.caption(
      "**Audit Advisory:** Always cross-reference extracted XBRL line items"
      " against official SEC 10-Q/10-K PDF filings for institutional accuracy."
  )


def calculate_fcf_from_raw(df):
  df_cf_raw = df[["Period", "OCF", "Capex"]].dropna(subset=["OCF"]).copy()
  df_cf_raw["Capex"] = df_cf_raw["Capex"].fillna(0)
  df_cf_raw["Month"] = pd.to_datetime(df_cf_raw["Period"]).dt.month
  standalone_ocf, standalone_capex = [], []
  prev_year, prev_ocf_ytd, prev_capex_ytd = None, 0, 0
  for idx, row in df_cf_raw.iterrows():
    m = row["Month"]
    curr_year = pd.to_datetime(row["Period"]).year
    ocf_ytd, capex_ytd = row["OCF"], row["Capex"]
    if curr_year != prev_year or m == 3:
      q_ocf, q_capex = ocf_ytd, capex_ytd
    else:
      q_ocf, q_capex = ocf_ytd - prev_ocf_ytd, capex_ytd - prev_capex_ytd
    standalone_ocf.append(q_ocf)
    standalone_capex.append(q_capex)
    prev_year = curr_year
    prev_ocf_ytd, prev_capex_ytd = ocf_ytd, capex_ytd

  df_fcf = pd.DataFrame({
      "Period": df_cf_raw["Period"],
      "FCF": np.array(standalone_ocf) - np.abs(np.array(standalone_capex)),
  })
  df_fcf["FCF_B"] = df_fcf["FCF"] / 1e9
  df_fcf["FCF_YoY_%"] = df_fcf["FCF"].pct_change(
      periods=4, fill_method=None
  ) * 100
  df_fcf["FCF_QoQ_%"] = df_fcf["FCF"].pct_change(
      periods=1, fill_method=None
  ) * 100
  return df_fcf


@st.cache_data(ttl=86400)
def fetch_and_parse_ticker(ticker):
  company = Company(ticker)
  filings_10q = company.get_filings(form="10-Q")[:40]

  def parse_multi_val(df, keywords):
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
              return float(val_str)
            except:
              pass
    return np.nan

  def get_exact_val(df, label_name):
    if df is None:
      return np.nan
    m = df[df["label"] == label_name]
    if m.empty:
      m = df[df["label"].str.contains(label_name, case=False, na=False)]
    if m.empty:
      return np.nan
    num_cols = [c for c in df.columns if "202" in str(c) or "201" in str(c)]
    if not num_cols:
      return np.nan
    val_raw = m.iloc[0][num_cols[0]]
    if pd.isna(val_raw):
      return np.nan
    val_str = (
        str(val_raw)
        .replace("$", "")
        .replace(",", "")
        .replace("(", "-")
        .replace(")", "")
    )
    try:
      return float(val_str)
    except:
      return np.nan

  records = []
  for f in filings_10q:
    obj = f.obj()
    inc = obj.income_statement
    cf = obj.cash_flow_statement
    if inc is None:
      continue
    inc_df = inc.to_dataframe(view="standard")
    cf_df = cf.to_dataframe(view="standard") if cf is not None else None

    rev = parse_multi_val(
        inc_df, ["Revenue", "Total revenue", "Revenues, net", "Net revenues"]
    )
    op_inc = parse_multi_val(
        inc_df,
        [
            "Income from operations",
            "Loss from operations",
            "Income (loss) from operations",
            "Operating income (loss)",
            "Profit from operations",
        ],
    )
    net_inc = parse_multi_val(
        inc_df,
        ["Net income including", "Net income (loss)", "Net loss", "Net income"],
    )
    eps_diluted = parse_multi_val(
        inc_df,
        [
            "Diluted",
            "Diluted earnings per share",
            "Earnings per share, diluted",
            "Diluted (in USD per share)",
            "Basic and diluted",
        ],
    )

    ocf = get_exact_val(cf_df, "Net cash provided by operating activities")
    capex = get_exact_val(cf_df, "Purchases of property and equipment")

    records.append({
        "Period": str(f.period_of_report),
        "Revenue": rev,
        "Operating_Income": op_inc,
        "Net_Income": net_inc,
        "Diluted_EPS": eps_diluted,
        "OCF": ocf,
        "Capex": capex,
    })

  df = pd.DataFrame(records).sort_values("Period").reset_index(drop=True)

  # Derived metrics
  df["Revenue_B"] = df["Revenue"] / 1e9
  df["Rev_YoY_%"] = df["Revenue_B"].pct_change(periods=4, fill_method=None) * 100
  df["Rev_QoQ_%"] = df["Revenue_B"].pct_change(periods=1, fill_method=None) * 100

  df["EPS_YoY_%"] = df["Diluted_EPS"].pct_change(
      periods=4, fill_method=None
  ) * 100
  df["EPS_QoQ_%"] = df["Diluted_EPS"].pct_change(
      periods=1, fill_method=None
  ) * 100

  df["Op_Margin_%"] = (df["Operating_Income"] / df["Revenue"]) * 100
  df["Net_Margin_%"] = (df["Net_Income"] / df["Revenue"]) * 100

  df_fcf = calculate_fcf_from_raw(df)
  return df, df_fcf


@st.cache_data(ttl=3600)
def fetch_market_data(ticker):
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


try:
  with st.spinner(f"Extracting SEC XBRL filings & market data for {ticker_symbol}..."):
    df_raw_full, df_fcf_full = fetch_and_parse_ticker(ticker_symbol)
    hist_price, current_price, market_cap = fetch_market_data(ticker_symbol)

  # Local slice based on slider
  df_raw = df_raw_full.tail(lookback_quarters).reset_index(drop=True)
  df_fcf = df_fcf_full.tail(lookback_quarters).reset_index(drop=True)

  # Calculate TTM & Valuation metrics per quarter
  if market_cap:
    df_raw["TTM_Revenue"] = df_raw["Revenue"].rolling(4).sum()
    df_raw["TTM_EPS"] = df_raw["Diluted_EPS"].rolling(4).sum()
    df_raw["P_S_Ratio"] = market_cap / df_raw["TTM_Revenue"]
    df_raw["P_E_Ratio"] = current_price / df_raw["TTM_EPS"]
    # Merge FCF into df_raw for FCF Yield
    merged_fcf = df_raw[["Period"]].merge(
        df_fcf[["Period", "FCF"]], on="Period", how="left"
    )
    df_raw["TTM_FCF"] = merged_fcf["FCF"].rolling(4).sum()
    df_raw["FCF_Yield_%"] = (df_raw["TTM_FCF"] / market_cap) * 100

  st.subheader(f"{ticker_symbol} — Executive 2x2 Financial Dashboard")

  fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
  fig.suptitle(
      f"{ticker_symbol} Comprehensive Financial & Growth Dashboard (Direct SEC"
      " XBRL)",
      fontsize=15,
      fontweight="bold",
      y=0.98,
  )

  # 1. Revenue
  ax1 = axes[0, 0]
  v_rev = df_raw.dropna(subset=["Revenue_B"])
  ax1.bar(
      v_rev["Period"],
      v_rev["Revenue_B"],
      color="#8B5CF6",
      alpha=0.85,
      width=0.55,
      label="Revenue ($B)",
  )
  ax1.set_title("Revenue ($B) & YoY/QoQ Growth", fontweight="bold", fontsize=10.5)
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

  # 4. Margins (Line Chart)
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
  st.subheader(f"{ticker_symbol} — Valuation & Price Action Dashboard")

  fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
  fig2.suptitle(
      f"{ticker_symbol} Valuation Multiples & Price Action (50/200 EMA)",
      fontsize=15,
      fontweight="bold",
      y=0.98,
  )

  # Valuation Chart 1: Stock Price + 50/200 EMA
  ax_p1 = axes2[0, 0]
  if not hist_price.empty:
    ax_p1.plot(
        hist_price.index,
        hist_price["Close"],
        color="#1F2937",
        linewidth=1.5,
        label="Close Price ($)",
    )
    ax_p1.plot(
        hist_price.index,
        hist_price["EMA50"],
        color="#3B82F6",
        linestyle="-",
        linewidth=1.2,
        label="50-Day EMA",
    )
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

  # Valuation Chart 2: P/S Ratio over Quarters
  ax_p2 = axes2[0, 1]
  ax_p2.plot(
      df_raw["Period"],
      df_raw["P_S_Ratio"],
      color="#8B5CF6",
      marker="o",
      linewidth=2,
      label="P/S (TTM)",
  )
  ax_p2.set_title("Price-to-Sales (P/S) Ratio", fontweight="bold", fontsize=10.5)
  ax_p2.set_ylabel("P/S Multiple (x)", color="#6D28D9")
  ax_p2.tick_params(axis="x", rotation=45, labelsize=7)
  ax_p2.legend(loc="upper left", fontsize=7)
  ax_p2.grid(True, linestyle="--", alpha=0.3)

  # Valuation Chart 3: P/E Ratio over Quarters
  ax_p3 = axes2[1, 0]
  ax_p3.plot(
      df_raw["Period"],
      df_raw["P_E_Ratio"],
      color="#10B981",
      marker="s",
      linewidth=2,
      label="P/E (TTM)",
  )
  ax_p3.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
  ax_p3.set_title("Price-to-Earnings (P/E) Ratio", fontweight="bold", fontsize=10.5)
  ax_p3.set_ylabel("P/E Multiple (x)", color="#047857")
  ax_p3.tick_params(axis="x", rotation=45, labelsize=7)
  axp3_legend = ax_p3.legend(loc="upper left", fontsize=7)
  ax_p3.grid(True, linestyle="--", alpha=0.3)

  # Valuation Chart 4: FCF Yield Line Graph
  ax_p4 = axes2[1, 1]
  ax_p4.plot(
      df_raw["Period"],
      df_raw["FCF_Yield_%"],
      color="#F59E0B",
      marker="^",
      linewidth=2,
      label="FCF Yield (%)",
  )
  ax_p4.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
  ax_p4.set_title("Free Cash Flow Yield (%)", fontweight="bold", fontsize=10.5)
  ax_p4.set_ylabel("FCF Yield (%)", color="#B45309")
  ax_p4.tick_params(axis="x", rotation=45, labelsize=7)
  ax_p4.legend(loc="upper left", fontsize=7)
  ax_p4.grid(True, linestyle="--", alpha=0.3)

  plt.tight_layout(rect=[0, 0, 1, 0.98])
  st.pyplot(fig2)

  with st.expander("View Raw Extracted Dataset & Valuations"):
    st.dataframe(df_raw, use_container_width=True)

except Exception as e:
  st.error(
      f"Could not load data for ticker '{ticker_symbol}'. Check ticker symbol or"
      f" network connection. Error: {e}"
  )
