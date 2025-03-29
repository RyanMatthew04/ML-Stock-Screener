import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import ta
import pickle


def rf():
    # Load saved RF model
    with open("pickle/rf_final.pkl", "rb") as f:
        rf_model = pickle.load(f)

    # Load the top 40 features used in training
    top_40_features = ['volatility_kcp',
    'trend_cci',
    'momentum_stoch_rsi',
    'momentum_stoch_rsi_k',
    'momentum_stoch',
    'momentum_wr',
    'volatility_bbp',
    'volume_em',
    'momentum_stoch_signal',
    'momentum_ppo_hist',
    'trend_vortex_ind_pos',
    'trend_adx_neg',
    'trend_macd_diff',
    'trend_vortex_ind_diff',
    'momentum_rsi',
    'trend_aroon_up',
    'trend_aroon_down',
    'trend_stc',
    'volatility_dcp',
    'trend_adx_pos',
    'momentum_roc',
    'momentum_stoch_rsi_d',
    'volatility_kcli',
    'volatility_ui',
    'trend_macd',
    'trend_psar_up_indicator',
    'volume_fi',
    'trend_vortex_ind_neg',
    'others_dlr',
    'trend_dpo',
    'momentum_tsi',
    'trend_aroon_ind',
    'trend_psar_down_indicator',
    'volatility_dcw',
    'volatility_bbw',
    'others_dr',
    'momentum_uo',
    'trend_kst',
    'volume_mfi',
    'trend_kst_sig']

    # Load instrument mappings (created earlier)
    with open("artifacts/NSE.json", "r") as f:
        instruments = json.load(f)
    instrument_df = pd.DataFrame([i for i in instruments if i['segment'] == 'NSE_EQ'])
    nifty_symbols = pd.read_csv("artifacts/nifty_50_symbols.csv")["Symbol"].tolist()
    instrument_df = instrument_df[instrument_df["trading_symbol"].isin(nifty_symbols)]
    symbol_map = instrument_df.set_index("trading_symbol")["instrument_key"].to_dict()

    # Date setup
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    # Fetch + TA calculation
    df_final = pd.DataFrame()
    for symbol, key in symbol_map.items():
        url = f"https://api.upstox.com/v2/historical-candle/{key}/day/{end_date}/{start_date}"
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        }

        try:
            res = requests.get(url, headers=headers)
            data = res.json()

            if 'data' in data and 'candles' in data['data']:
                candles = pd.DataFrame(data['data']['candles'],
                                    columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                candles = candles.drop(columns=['OI'])
                candles['Timestamp'] = pd.to_datetime(candles['Timestamp'])
                candles = candles.sort_values(by='Timestamp')

                # Set index for TA
                candles.set_index('Timestamp', inplace=True)

                # Apply TA
                df_ta = ta.add_all_ta_features(
                    candles, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
                )

                # Keep only last row (latest day)
                latest = df_ta.iloc[[-1]].copy()
                latest["Symbol"] = symbol
                df_final = pd.concat([df_final, latest], ignore_index=True)
            else:
                print(f"No data for {symbol}")
            time.sleep(0.3)
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            continue

    # Filter top 40 features
    X_latest = df_final[top_40_features]

    # Predict probabilities
    probs = rf_model.predict_proba(X_latest)

    # Prepare output
    results = pd.DataFrame({
        "Stock": df_final["Symbol"],
        "RF_Bearish_Probability": probs[:, 0],
        "RF_Bullish_Probability": probs[:, 1]
    })
    return results
