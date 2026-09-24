"""Explain session win-rate denominators and independently reconcile trade P&L."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    args = parser.parse_args()
    records, portfolios = [], {}
    for path in sorted(args.comparison_dir.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        portfolio = pd.DataFrame(summary["daily_portfolio"]).set_index("entry_date")
        trades = pd.read_csv(path.parent / "trades.csv")
        groups = trades.groupby("entry_date")
        returns = portfolio.strategy_return
        active = portfolio.traded
        # Reconcile reported returns with actual quantities, prices and financing.
        pnl = (
            trades.quantity * (trades.exit_price - trades.entry_price)
            - trades.transaction_cost_dollars
        ).groupby(trades.entry_date).sum().reindex(portfolio.index, fill_value=0)
        independent = (pnl - portfolio.borrow_cost) / portfolio.portfolio_start_equity
        np.testing.assert_allclose(independent, returns, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            groups.net_return.mean(), portfolio.loc[active, "unscaled_return"],
            rtol=0, atol=1e-12,
        )
        config = summary["strategy_config"]
        count = config.get("allocation_count", summary["top"])
        assert groups.size().eq(count).all()
        weights = trades.entry_notional / groups.entry_notional.transform("sum")
        np.testing.assert_allclose(weights, 1 / count, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            (returns > 0).mean(), summary["strategy_metrics"]["win_rate"],
            rtol=0, atol=1e-12,
        )
        assert returns[~active].eq(0).all()
        portfolios[summary["strategy"]] = portfolio
        records.append({
            "strategy": summary["strategy"], "sessions": len(portfolio),
            "traded_sessions": int(active.sum()), "cash_sessions": int((~active).sum()),
            "winning_sessions": int((returns > 0).sum()),
            "losing_sessions": int((returns < 0).sum()),
            "flat_traded_sessions": int((returns[active] == 0).sum()),
            "reported_win_rate": summary["strategy_metrics"]["win_rate"],
            "traded_session_win_rate": float((returns[active] > 0).mean()),
            "individual_stock_win_rate_before_financing": float((trades.net_return > 0).mean()),
            "total_return": summary["strategy_metrics"]["total_return"],
            "cash_sessions_by_month": portfolio.loc[~active].groupby(
                portfolio.loc[~active].index.str[:7]
            ).size().to_dict(),
            "pnl_and_equal_weight_checks": "passed within 1e-12",
        })
    focus = portfolios["liquidity-momentum-focus"]
    trend = portfolios["liquidity-trend-vol"]
    pd.testing.assert_index_equal(focus.index, trend.index)
    common = focus.traded & trend.traded
    missed = ~focus.traded & trend.traded
    report = {
        "results": records,
        "same_traded_sessions": {
            "count": int(common.sum()),
            "focus_wins": int((focus.loc[common, "strategy_return"] > 0).sum()),
            "trend_vol_wins": int((trend.loc[common, "strategy_return"] > 0).sum()),
        },
        "trend_vol_on_focus_cash_sessions": {
            "count": int(missed.sum()),
            "wins": int((trend.loc[missed, "strategy_return"] > 0).sum()),
            "losses": int((trend.loc[missed, "strategy_return"] < 0).sum()),
            "compounded_return": float(np.prod(1 + trend.loc[missed, "strategy_return"]) - 1),
        },
    }
    output = args.comparison_dir / "win_rate_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Audit: {output.resolve()}")


if __name__ == "__main__":
    main()
