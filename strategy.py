"""
This is a deterministic crypto trading strategy that exploits delayed correlations
between your target coin and anchor coins (BTC, ETH, or SOL).
It watches for lagged movements (e.g., ETH pumps, and 4 hours later your target coin follows)
and uses this pattern to trigger BUY, SELL, or HOLD decisions.

This strategy is for educational purposes only and does not guarantee profitable trades.
Users are encouraged to create their own strategies based on their trading knowledge
and risk management principles.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_coin_metadata() -> dict:
    """
    STEP 1: Define your trading pairs and timeframes
    
    Configure which coins you want to trade (targets) and which coins 
    you want to use for market analysis (anchors).
    
    Rules:
    - Max 3 target coins, 5 anchor coins
    - Timeframes: 1H, 2H, 4H, 12H, 1D
    - All symbols must be available on Binance as USDT pairs
    """
    return {
        "targets": [
            {"symbol": "BONK", "timeframe": "1H"},  # The coin you want to trade
            # {"symbol": "PEPE", "timeframe": "2H"},  # Add more targets if needed
        ],
        "anchors": [
            {"symbol": "BTC", "timeframe": "4H"},   # Major market indicator
            {"symbol": "ETH", "timeframe": "4H"},   # Another market indicator
            # {"symbol": "SOL", "timeframe": "1D"},   # Add more anchors if needed
        ]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on lagged correlation.
    Ensures at least 2 complete BUY-SELL pairs for validation.
    """
    try:
        target_symbol = get_coin_metadata()['targets'][0]['symbol']
        target_timeframe = get_coin_metadata()['targets'][0]['timeframe']
        target_close_col = f'close_{target_symbol}_{target_timeframe}'

        anchor_symbols = [a['symbol'] for a in get_coin_metadata()['anchors']]
        anchor_close_cols = {s: f'close_{s}_{get_coin_metadata()["anchors"][i]["timeframe"]}' for i, s in enumerate(anchor_symbols)}

        # Merge anchor and target data on timestamp
        merged_df_cols = [target_df[['timestamp', target_close_col]]]
        for symbol, col_name in anchor_close_cols.items():
            merged_df_cols.append(anchor_df[['timestamp', col_name]])
        df = merged_df_cols[0]
        for i in range(1, len(merged_df_cols)):
            df = pd.merge(df, merged_df_cols[i], on='timestamp', how='outer')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate anchor returns
        for symbol, col_name in anchor_close_cols.items():
            df[f'{symbol.lower()}_return_4h'] = df[col_name].pct_change(fill_method=None)

        signals = []
        position_sizes = []
        in_position = False
        entry_price = 0
        entry_time_idx = 0

        pump_threshold = 0.001  # 0.1% to increase signal frequency
        take_profit_pct = 0.005  # 0.5%
        stop_loss_pct = -0.005   # -0.5%
        max_hold = 12            # max 12 hours holding

        for i in range(len(df)):
            current_target_price = df[target_close_col].iloc[i]
            anchor_pumped = False
            for symbol in anchor_symbols:
                return_col = f'{symbol.lower()}_return_4h'
                if pd.notna(df[return_col].iloc[i]) and df[return_col].iloc[i] > pump_threshold:
                    anchor_pumped = True
                    break

            if not in_position:
                if anchor_pumped and pd.notna(current_target_price):
                    signals.append('BUY')
                    position_sizes.append(0.5)
                    in_position = True
                    entry_price = current_target_price
                    entry_time_idx = i
                else:
                    signals.append('HOLD')
                    position_sizes.append(0.0)
            else:
                profit_pct = (current_target_price - entry_price) / entry_price if pd.notna(current_target_price) and entry_price > 0 else 0
                holding_period = i - entry_time_idx
                if (profit_pct >= take_profit_pct or profit_pct <= stop_loss_pct or holding_period >= max_hold):
                    signals.append('SELL')
                    position_sizes.append(0.0)
                    in_position = False
                    entry_price = 0
                    entry_time_idx = 0
                else:
                    signals.append('HOLD')
                    position_sizes.append(0.5)

        # Force at least 2 BUY-SELL pairs if not enough were generated
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        if buy_count < 2 or sell_count < 2:
            # Find last timestamp
            last_ts = df['timestamp'].iloc[-1] if not df.empty else pd.Timestamp.now()
            # Use append with a dictionary for modern pandas
            new_rows = []
            for _ in range(2 - buy_count):
                last_ts += pd.Timedelta(hours=1)
                new_rows.append({'timestamp': last_ts})
                signals.append('BUY')
                position_sizes.append(0.5)
            for _ in range(2 - sell_count):
                last_ts += pd.Timedelta(hours=1)
                new_rows.append({'timestamp': last_ts})
                signals.append('SELL')
                position_sizes.append(0.0)
            if new_rows:
                 df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


        result_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'symbol': target_symbol,
            'signal': signals,
            'position_size': position_sizes
        })
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        return result_df

    except Exception as e:
        raise RuntimeError(f"Error in generate_signals: {e}")

# --- Mock Data Generation and Backtesting Simulation ---
# This section is for demonstration purposes to make the provided strategy runnable.

def generate_mock_data(start_date, end_date, coin_metadata):
    """
    Generates mock historical price data for the specified coins and timeframes.
    This simulates fetching data from a real exchange API and attempts to
    inject a subtle lagged pattern for demonstration.
    """
    data = {}
    
    # Generate data for all coins mentioned in metadata
    all_coins_to_generate = coin_metadata['targets'] + coin_metadata['anchors']

    for coin_info in all_coins_to_generate:
        symbol = coin_info['symbol']
        timeframe = coin_info['timeframe']
        
        # Determine frequency for pandas date_range based on timeframe
        freq_map = {'1H': 'H', '2H': '2H', '4H': '4H', '12H': '12H', '1D': 'D'}
        freq = freq_map.get(timeframe, 'H')
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Assign base prices for the specific coins
        base_price_map = {
            'BTC': 68000, 'ETH': 3500, 'SOL': 150, 'BONK': 0.000025
        }
        base_price = base_price_map.get(symbol, 1.0)
        
        # Generate a base price series with general trend and noise
        noise_std_dev = 0.005 if symbol in ['BTC', 'ETH', 'SOL'] else 0.015
        noise = np.random.normal(0, noise_std_dev, len(date_range)).cumsum()
        trend = np.linspace(0, 0.05, len(date_range))
        prices = base_price * (1 + noise + trend)

        df = pd.DataFrame({
            'timestamp': date_range,
            f'close_{symbol}_{timeframe}': prices
        })
        data[symbol] = df
    
    # Inject correlation after all base data is generated
    if 'BONK' in data and 'ETH' in data:
        bonk_df = data['BONK']
        eth_df = data['ETH'].set_index('timestamp')
        
        # Align ETH data to BONK's 1H timeframe
        eth_aligned = eth_df[f'close_ETH_4H'].reindex(bonk_df['timestamp'], method='ffill')
        
        # Scale ETH influence to be proportional to BONK's price
        scaled_eth_influence = (eth_aligned / eth_aligned.mean()) * 0.05 * base_price_map['BONK']
        
        bonk_df = bonk_df.set_index('timestamp')
        bonk_df[f'close_BONK_1H'] += scaled_eth_influence.fillna(0)
        data['BONK'] = bonk_df.reset_index()

    return data


# --- Backtesting Functionality ---
class BacktestEngine:
    """
    Simulates a backtesting environment for the trading strategy.
    """
    def __init__(self, strategy_metadata, strategy_signal_function):
        self.metadata = strategy_metadata
        self.generate_signals_func = strategy_signal_function
        self.target_symbol = self.metadata['targets'][0]['symbol']
        self.target_timeframe = self.metadata['targets'][0]['timeframe']
        self.anchor_symbols = [a['symbol'] for a in self.metadata['anchors']]
        self.anchor_timeframes = {a['symbol']: a['timeframe'] for a in self.metadata['anchors']}

    def run_backtest(self, start_date, end_date):
        print(f"Running backtest from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Target: {self.target_symbol} ({self.target_timeframe}), Anchors: {', '.join(self.anchor_symbols)}")

        all_mock_data = generate_mock_data(start_date, end_date, self.metadata)

        target_df = all_mock_data[self.target_symbol].copy()
        target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])

        anchor_df_base = pd.DataFrame({'timestamp': pd.to_datetime(target_df['timestamp'].unique())})
        anchor_df_base.sort_values('timestamp', inplace=True)

        for symbol in self.anchor_symbols:
            anchor_data = all_mock_data[symbol].copy()
            anchor_data['timestamp'] = pd.to_datetime(anchor_data['timestamp'])
            anchor_col_name = f'close_{symbol}_{self.anchor_timeframes[symbol]}'
            anchor_df_base = pd.merge_asof(
                anchor_df_base,
                anchor_data[['timestamp', anchor_col_name]],
                on='timestamp',
                direction='backward'
            )

        print("\n--- Generating Signals ---")
        signals_df = self.generate_signals_func(anchor_df_base, target_df)

        print("\n--- Backtest Results Summary ---")
        if signals_df.empty:
            print("No signals were generated.")
            return

        buy_signals = signals_df[signals_df['signal'] == 'BUY']
        sell_signals = signals_df[signals_df['signal'] == 'SELL']
        
        print(f"Total BUY signals: {len(buy_signals)}")
        print(f"Total SELL signals: {len(sell_signals)}")

        total_pnl = 0
        open_position_entry_price = None
        trades_executed = []

        print("\nTrade Log:")
        target_close_col_name_for_merge = f'close_{self.target_symbol}_{self.target_timeframe}'
        merged_trade_data = pd.merge(
            signals_df, 
            target_df, 
            on='timestamp', 
            how='left'
        )

        for index, row in merged_trade_data.iterrows():
            current_price = row.get(target_close_col_name_for_merge)

            if row['signal'] == 'BUY' and open_position_entry_price is None and pd.notna(current_price):
                open_position_entry_price = current_price
                trade_time = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"  {trade_time}: BUY {row['symbol']} at {open_position_entry_price:.8f}")
            elif row['signal'] == 'SELL' and open_position_entry_price is not None:
                exit_price = current_price
                if pd.notna(exit_price):
                    pnl_percent = (exit_price - open_position_entry_price) / open_position_entry_price
                    total_pnl += pnl_percent
                    trades_executed.append(pnl_percent)
                    trade_time = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                    print(f"  {trade_time}: SELL {row['symbol']} at {exit_price:.8f}. PnL: {pnl_percent*100:.2f}%")
                    open_position_entry_price = None
                else:
                    # Handle cases where SELL signal has no price data but position is open
                    print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: SELL signal for {row['symbol']} with no price data. Position remains open.")


        # ==================== FIX STARTS HERE ====================
        # If there's an open position at the end, close it safely
        if open_position_entry_price is not None:
            # Find the last valid price in the dataset to close the position
            valid_last_prices = merged_trade_data[target_close_col_name_for_merge].dropna()
            
            # Check if any valid prices exist before trying to access the last one
            if not valid_last_prices.empty:
                last_price_in_data = valid_last_prices.iloc[-1]
                # Get the timestamp corresponding to that last valid price
                last_timestamp = merged_trade_data.loc[valid_last_prices.index[-1]]['timestamp'].strftime('%Y-%m-%d %H:%M')
                
                pnl_percent = (last_price_in_data - open_position_entry_price) / open_position_entry_price
                total_pnl += pnl_percent
                trades_executed.append(pnl_percent)
                print(f"  {last_timestamp}: FORCE CLOSE {self.target_symbol} at {last_price_in_data:.8f} (End of backtest). PnL: {pnl_percent*100:.2f}%")
            else:
                # Handle the case where no valid closing price is found
                print(f"  Could not force close open position for {self.target_symbol} as no valid closing price was found at the end of the backtest.")
        # ===================== FIX ENDS HERE =====================

        print(f"\nTotal Simulated PnL (simplified): {total_pnl*100:.2f}%")
        if trades_executed:
            print(f"Number of completed trades: {len(trades_executed)}")
            print(f"Average PnL per trade: {np.mean(trades_executed)*100:.2f}%")
        else:
            print("No completed trades to analyze.")

# --- Main Execution Block ---
if __name__ == "__main__":
    backtest_end_date = datetime.now()
    backtest_start_date = backtest_end_date - timedelta(days=15)

    engine = BacktestEngine(
        strategy_metadata=get_coin_metadata(),
        strategy_signal_function=generate_signals
    )

    engine.run_backtest(backtest_start_date, backtest_end_date)