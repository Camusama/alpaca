# liquidate existing positions
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass


def get_unrealized_pl(underlying_symbol: str, trading_client: TradeClient):
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


get_unrealized_pl()
