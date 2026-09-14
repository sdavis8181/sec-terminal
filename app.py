def calculate_fcf_from_raw(df):
  df_cf_raw = df[["Period", "OCF", "Capex"]].copy()
  df_cf_raw["OCF"] = pd.to_numeric(df_cf_raw["OCF"], errors="coerce")
  df_cf_raw["Capex"] = pd.to_numeric(df_cf_raw["Capex"], errors="coerce")

  # Sort chronologically just in case
  df_cf_raw = df_cf_raw.sort_values("Period").reset_index(drop=True)

  standalone_ocf, standalone_capex = [], []
  prev_year = None
  prev_ocf_ytd, prev_capex_ytd = 0.0, 0.0

  for idx, row in df_cf_raw.iterrows():
    p_date = pd.to_datetime(row["Period"])
    curr_year = p_date.year
    month = p_date.month

    ocf_val = row["OCF"] if pd.notna(row["OCF"]) else 0.0
    capex_val = row["Capex"] if pd.notna(row["Capex"]) else 0.0

    # Q1 (approx month 3 or new fiscal year start) represents 3-month standalone
    if curr_year != prev_year or month <= 4:
      q_ocf = ocf_val
      q_capex = capex_val
    else:
      # Subtract previous YTD from current YTD
      q_ocf = ocf_val - prev_ocf_ytd
      q_capex = capex_val - prev_capex_ytd
      # If subtraction results in an obvious negative anomaly from a restatement/gap, fallback to raw
      if q_ocf < -1e9:
        q_ocf = ocf_val

    standalone_ocf.append(q_ocf)
    standalone_capex.append(q_capex)
    prev_year = curr_year
    prev_ocf_ytd = ocf_val
    prev_capex_ytd = capex_val

  df_fcf = pd.DataFrame({
      "Period": df_cf_raw["Period"],
      "FCF": np.array(standalone_ocf) - np.abs(np.array(standalone_capex)),
  })
  df_fcf["FCF_B"] = df_fcf["FCF"] / 1e9
  df_fcf["FCF_YoY_%"] = (
      df_fcf["FCF"].pct_change(periods=4, fill_method=None).clip(-200, 200) * 100
  )
  df_fcf["FCF_QoQ_%"] = (
      df_fcf["FCF"].pct_change(periods=1, fill_method=None).clip(-200, 200) * 100
  )
  return df_fcf
