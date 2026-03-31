"""
close_all_positions.py — Demo reset utility
Closes every open position in your Alpaca paper account so the live demo
starts from cash and you can watch BUY signals fire from scratch.

Usage:
    python close_all_positions.py          # preview (dry run, no orders sent)
    python close_all_positions.py --confirm # actually close everything
"""

import os
import sys
import argparse
from alpaca_trade_api.rest import REST

def main():
    parser = argparse.ArgumentParser(description='Close all Alpaca paper positions (demo reset)')
    parser.add_argument('--confirm', action='store_true',
                        help='Actually submit close orders (omit for a dry-run preview)')
    args = parser.parse_args()

    api = REST(
        os.environ['APCA_API_KEY_ID'],
        os.environ['APCA_API_SECRET_KEY'],
        os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
    )

    try:
        positions = api.list_positions()
    except Exception as e:
        print(f"  Could not fetch positions: {e}")
        sys.exit(1)

    if not positions:
        print("  No open positions — account is already flat. Ready for demo.")
        return

    label = 'DRY RUN' if not args.confirm else 'CLOSING'
    print(f"\n  {label} -- {len(positions)} open position(s):\n")
    print(f"  {'TICKER':<8} {'QTY':>6}  {'ENTRY':>8}  {'CURRENT':>8}  {'P&L':>8}")
    print(f"  {'-'*8} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")

    for pos in positions:
        entry   = float(pos.avg_entry_price)
        current = float(pos.current_price)
        pnl     = float(pos.unrealized_pl)
        print(f"  {pos.symbol:<8} {pos.qty:>6}  ${entry:>7.2f}  ${current:>7.2f}  "
              f"{'+'if pnl>=0 else ''}{pnl:>7.2f}")

    print()

    if not args.confirm:
        print("  -- DRY RUN: no orders sent. Add --confirm to actually close. --")
        return

    print("  Closing all positions and cancelling open orders...")
    try:
        api.cancel_all_orders()
        print("  [OK] All pending orders cancelled.")
    except Exception as e:
        print(f"  Warning: could not cancel orders: {e}")

    # Close each position individually — works outside market hours in paper trading
    failed = []
    for pos in positions:
        try:
            api.close_position(pos.symbol)
            print(f"  [OK] {pos.symbol} closed.")
        except Exception as e:
            print(f"  [FAIL] {pos.symbol}: {e}")
            failed.append(pos.symbol)

    if failed:
        print(f"\n  Could not close: {', '.join(failed)}")
        print("  Tip: reset via Alpaca dashboard -> Paper Trading -> Reset Paper Account")
        sys.exit(1)
    else:
        print("\n  [OK] All positions closed. Account is now flat.")
        print("  [OK] Ready for demo -- run: python main.py --demo --interval 2")

if __name__ == '__main__':
    main()
