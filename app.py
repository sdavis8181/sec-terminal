rev = get_val(
            q_inc, [
                "Total Revenue",
                "Total Revenues",
                "Operating Revenue",
                "Revenue",
            ]
        )
        op_inc = get_val(q_inc, ["Operating Income", "EBIT", "Operating Profit"])
        net_inc = get_val(
            q_inc, ["Net Income", "Net Income Common Stockholders"]
        )
        eps_dil = get_val(
            q_inc, [
                "Diluted EPS",
                "Basic EPS",
                "Diluted Earnings Per Share",
                "Basic EPS",
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
        if pd.isna(shares):
          shares = get_val(
              q_inc, [
                  "Diluted Average Shares",
                  "Basic Average Shares",
                  "Average Diluted Shares Outstanding",
              ]
          )
