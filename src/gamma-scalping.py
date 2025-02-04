# %%
# Please change the following to your own PAPER api key and secret
# or set them as environment variables (ALPACA_API_KEY, ALPACA_SECRET_KEY).
# You can get them from https://alpaca.markets/
import alpaca
from alpaca.trading.enums import AssetStatus, ContractType, AssetClass
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.models import TradeUpdate
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.historical.option import OptionHistoricalDataClient, OptionLatestQuoteRequest
import nest_asyncio
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np
import pandas as pd
import asyncio
import time
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import (
    AssetStatus,
    ExerciseStyle,
    OrderSide,
    OrderType,
    TimeInForce,
    QueryOrderStatus
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    GetAssetsRequest,
    MarketOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest
)
from alpaca.data.requests import (
    OptionBarsRequest,
    OptionTradesRequest,
    OptionLatestQuoteRequest,
    OptionLatestTradeRequest,
    OptionSnapshotRequest,
    OptionChainRequest
)
from alpaca.data.live.option import OptionDataStream
from alpaca.trading.stream import TradingStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
api_key = None
secret_key = None
TRADE_API_KEY = api_key
TRADE_API_SECRET = secret_key

#### We use paper environment for this example ####
# Please do not modify this. This example is for paper trading only.
paper = True
####

# Below are the variables for development this documents
# Please do not change these variables

trade_api_url = None
trade_api_wss = None
data_api_url = None
option_stream_data_wss = None

# %%

if api_key is None:
    TRADE_API_KEY = os.environ.get('ALPACA_API_KEY')

if secret_key is None:
    TRADE_API_SECRET = os.environ.get('ALPACA_SECRET_KEY')

# %%
# install alpaca-py if it is not available


# %%


# to run async code in jupyter notebook
nest_asyncio.apply()
# check version of alpaca-py
alpaca.__version__

# %%
# Initialize Alpaca clients

trading_client = TradingClient(
    api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET, paper=paper)
trade_update_stream = TradingStream(
    api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET, paper=paper)
stock_data_client = StockHistoricalDataClient(
    api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET)
option_data_client = OptionHistoricalDataClient(
    api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET)

# %%
# Configuration

underlying_symbol = "NVDA"
max_abs_notional_delta = 200
risk_free_rate = 0.045
positions = {}
stock_trades = {
    'avg_price': 0.0,
    'total_shares': 0.0,
    'realized_pnl': 0.0
}

# %%
# liquidate existing positions

print(
    f"Liquidating pre-existing positions related to underlying {underlying_symbol}")
all_positions = trading_client.get_all_positions()

for p in all_positions:
    if p.asset_class == AssetClass.US_OPTION:
        option_contract = trading_client.get_option_contract(p.symbol)
        if option_contract.underlying_symbol == underlying_symbol:
            print(f"Liquidating {p.qty} of {p.symbol}")
            trading_client.close_position(p.symbol)
    elif p.asset_class == AssetClass.US_EQUITY:
        if p.symbol == underlying_symbol:
            print(f"Liquidating {p.qty} of {p.symbol}")
            trading_client.close_position(p.symbol)

# %%
# Add underlying symbol to positions list

print(f"Adding {underlying_symbol} to position list")
positions[underlying_symbol] = {
    'asset_class': 'us_equity', 'position': 0, 'initial_position': 0}
# Set expiration range for options

today = datetime.now().date()
min_expiration = today + timedelta(days=14)
max_expiration = today + timedelta(days=60)


# %%
# Get the latest price of the underlying stock

def get_underlying_price(symbol):

    underlying_trade_request = StockLatestTradeRequest(
        symbol_or_symbols=symbol)
    underlying_trade_response = stock_data_client.get_stock_latest_trade(
        underlying_trade_request)
    return underlying_trade_response[symbol].price


underlying_price = get_underlying_price(underlying_symbol)
min_strike = round(underlying_price * 1.01, 2)

print(f"{underlying_symbol} price: {underlying_price}")
print(
    f"Min Expiration: {min_expiration}, Max Expiration: {max_expiration}, Min Strike: {min_strike}")

# %%
# Search for option contracts to add to the portfolio

req = GetOptionContractsRequest(
    underlying_symbols=[underlying_symbol],
    status=AssetStatus.ACTIVE,
    expiration_date_gte=min_expiration,
    expiration_date_lte=max_expiration,
    root_symbol=underlying_symbol,
    type=ContractType.CALL,
    strike_price_gte=str(min_strike),
    limit=5,
)

option_chain_list = trading_client.get_option_contracts(req).option_contracts

# %%
# Add the first 3 options to the position list

for option in option_chain_list[:3]:
    symbol = option.symbol
    print(f"Adding {symbol} to position list", option)
    positions[symbol] = {
        'asset_class': 'us_option',
        'underlying_symbol': option.underlying_symbol,
        'expiration_date': pd.Timestamp(option.expiration_date),
        'strike_price': float(option.strike_price),
        'type': option.type,
        'size': float(option.size),
        'position': 1.0,
        'initial_position': 1.0,
        'name': option.name,
    }

# %%
# Calculate implied volatility


def calculate_implied_volatility(option_price, S, K, T, r, option_type):
    def option_price_diff(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price - option_price

    return brentq(option_price_diff, 1e-6, 1)

# %%
# Calculate option Greeks (Delta and Gamma)


def calculate_greeks(option_price, strike_price, expiry, underlying_price, risk_free_rate, option_type):
    T = (expiry - pd.Timestamp.now()).days / 365
    implied_volatility = calculate_implied_volatility(
        option_price, underlying_price, strike_price, T, risk_free_rate, option_type)
    d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 *
          implied_volatility ** 2) * T) / (implied_volatility * np.sqrt(T))
    d2 = d1 - implied_volatility * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(T))
    return delta, gamma

# %%


def get_unrealized_pl():
    print(f"get_unrealized_pl {underlying_symbol}")
    equity_pl = 0.0
    option_pl = 0.0
    all_positions = trading_client.get_all_positions()
    # print(all_positions)
    for p in all_positions:
        if (p.symbol != underlying_symbol):
            continue
        if p.asset_class == AssetClass.US_OPTION:
            option_pl += float(p.unrealized_pl)
        elif p.asset_class == AssetClass.US_EQUITY or p.asset_class == AssetClass.CRYPTO:
            equity_pl = float(p.unrealized_pl)
    print(f"Equity P&L: ${equity_pl:.2f}")
    print(f"Option P&L: ${option_pl:.2f}")
    print(f"Total P&L: ${equity_pl + option_pl:.2f}")


# %%
# Handle trade updates
hasInitOption = False


async def on_trade_updates(data: TradeUpdate):
    symbol = data.order.symbol
    if symbol in positions:
        if data.event in {'fill', 'partial_fill'}:
            side = data.order.side
            qty = float(data.order.qty)
            filled_avg_price = float(data.order.filled_avg_price)
            position_qty = data.position_qty
            print(f"{data.event} event: {side} {qty} {symbol} @ {filled_avg_price}")
            print(
                f"updating position from {positions[symbol]['position']} to {position_qty}")
            positions[symbol]['position'] = float(position_qty)
            if symbol == underlying_symbol and data.order.asset_class == AssetClass.US_OPTION:
                hasInitOption = True
            if symbol == underlying_symbol and data.order.asset_class == AssetClass.US_EQUITY:
                # 更新股票交易记录
                if side == 'sell':
                    # 做空时更新平均成本
                    old_total = stock_trades['avg_price'] * \
                        stock_trades['total_shares']
                    new_total = filled_avg_price * (-qty)
                    stock_trades['total_shares'] -= qty
                    stock_trades['avg_price'] = (
                        old_total + new_total) / stock_trades['total_shares']
                else:  # buy to cover
                    # 计算已实现盈亏
                    profit = (stock_trades['avg_price'] -
                              filled_avg_price) * qty
                    stock_trades['realized_pnl'] += profit
                    # 更新剩余仓位的成本基础
                    stock_trades['total_shares'] += qty
                    if stock_trades['total_shares'] != 0:
                        old_total = stock_trades['avg_price'] * \
                            stock_trades['total_shares']
                        new_total = filled_avg_price * (qty)
                        stock_trades['avg_price'] = (
                            old_total + new_total) / stock_trades['total_shares']
                print(f"Average Cost: ${stock_trades['avg_price']:.2f}")
                print(f"Total Share:  ${stock_trades['total_shares']:.2f}")
                print(f"Realized P&L: ${stock_trades['realized_pnl']:.2f}")
                get_unrealized_pl()


trade_update_stream.subscribe_trade_updates(on_trade_updates)

# %%


def print_trades():
    print(f"Average Cost: ${stock_trades['avg_price']:.2f}")
    print(f"Total Share:  ${stock_trades['total_shares']:.2f}")
    print(f"Realized P&L: ${stock_trades['realized_pnl']:.2f}")


print_trades()

# %%
# Execute initial trades


async def initial_trades():
    await asyncio.sleep(5)
    print("executing initial option trades", positions.items())
    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_option' and pos['initial_position'] != 0:
            side = 'buy' if pos['initial_position'] > 0 else 'sell'
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(pos['initial_position']),
                side=side,
                type='market',
                time_in_force='day'
            )
            print(
                f"Submitting order to {side} {abs(pos['initial_position'])} contracts of {symbol} at market")
            trading_client.submit_order(order_request)

# %%
# Maintain delta-neutral strategy


def maintain_delta_neutral():
    current_delta = 0.0
    underlying_price = get_underlying_price(underlying_symbol)

    print(f"Current price of {underlying_symbol} is {underlying_price}")

    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_equity' and symbol == underlying_symbol:
            current_delta += pos['position']
        elif pos['asset_class'] == 'us_option' and pos['underlying_symbol'] == underlying_symbol:
            option_quote_request = OptionLatestQuoteRequest(
                symbol_or_symbols=symbol)
            option_quote = option_data_client.get_option_latest_quote(option_quote_request)[
                symbol]
            option_quote_mid = (option_quote.bid_price +
                                option_quote.ask_price) / 2

            delta, gamma = calculate_greeks(
                option_price=option_quote_mid,
                strike_price=pos['strike_price'],
                expiry=pos['expiration_date'],
                underlying_price=underlying_price,
                risk_free_rate=risk_free_rate,
                option_type=pos['type']
            )

            current_delta += delta * pos['position'] * pos['size']
    # try:
    #     adjust_delta(current_delta, underlying_price)
    # except:
    #     raise Exception(f"{underlying_symbol} maintain_delta_neutral")
    adjust_delta(current_delta, underlying_price)


def adjust_delta(current_delta, underlying_price):
    if current_delta * underlying_price > max_abs_notional_delta:
        side = 'sell'
    elif current_delta * underlying_price < -max_abs_notional_delta:
        side = 'buy'
    else:
        return

    qty = abs(round(current_delta, 0))
    order_request = MarketOrderRequest(
        symbol=underlying_symbol, qty=qty, side=side, type='market', time_in_force='day')
    if hasInitOption == False:
        print(f"{underlying_symbol} no options, return")
        raise Exception(f"{underlying_symbol} no options, return")
        return
    print(
        f"Submitting {side} order for {qty} shares of {underlying_symbol} at market")
    trading_client.submit_order(order_request)

# %%
# Gamma scalping strategy


async def gamma_scalp(initial_interval=10, interval=120):
    running = True
    await asyncio.sleep(initial_interval)

    maintain_delta_neutral()
    while running == True:
        try:
            await asyncio.sleep(interval)
            print("gamma_scalp loop")
            maintain_delta_neutral()
        except:
            print("gamma_scalp no options, return")
            running = False

# %%
# Main event loop https://alpaca.markets/learn/gamma-scalping

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(
    trade_update_stream._run_forever(),
    initial_trades(),
    gamma_scalp()
))
loop.close()
