import itertools
import unittest

import numpy as np
import pandas as pd

from trading_rl.overnight import backtest, live, ranking
from overnight.tests.test_strategy_export import ranking_inputs, replay_inputs


class SharedRankingTest(unittest.TestCase):
    def test_all_callers_share_membership_scores_and_history_eligibility(self):
        for scheme in ("dollar_ema", "turnover_stability"):
            with self.subTest(scheme=scheme):
                inputs = ranking_inputs()
                inputs.update(liquidity_scheme=scheme, entry_price_source="nbbo-ask")
                # Missing observations and a newly listed symbol must be treated
                # identically by the matrix and live bar-record adapters.
                inputs["dollar_volume"][3, 3] = np.nan
                inputs["dollar_volume"][:12, 5] = np.nan
                dates, symbols = inputs["dates"], inputs["symbols"]
                raw = {
                    symbol: [
                        {"t": f"{day.date()}T04:00:00Z", "v": float(value), "vw": 1., "c": 1.}
                        for day, value in zip(dates, inputs["dollar_volume"][:, column])
                        if np.isfinite(value)
                    ]
                    for column, symbol in enumerate(symbols)
                }
                exported = ranking.replay_strategy_selections(**replay_inputs(inputs))
                trades, _ = backtest.run_backtest(
                    **inputs, morning_prices=inputs["entry_prices"],
                    morning_staleness=inputs["entry_staleness"],
                    transaction_cost_bps=0., max_exit_staleness_minutes=1,
                )
                self.assertEqual(exported.sample_id.tolist(), trades.sample_id.tolist())
                completed = ranking.causal_completed_trading_days(inputs["dollar_volume"])
                for day, selected in exported.groupby("entry_date", sort=True):
                    index = dates.get_loc(pd.Timestamp(day))
                    available = {
                        name: bars for column, (name, bars) in enumerate(raw.items())
                        if name != "SPY" and inputs["execution_exchange_mask"][index + 1, column]
                    }
                    live_ranking = live.completed_liquidity_ranking(
                        available, [stamp.date() for stamp in dates], pd.Timestamp(day).date(),
                        inputs["ema_span"], inputs["min_history_days"], inputs["minimum_trading_days"], scheme,
                    )
                    candidates = [{"symbol": symbol, "score": score, "issuer": inputs["issuers"].get(symbol, symbol)}
                                  for symbol, score, _ in live_ranking]
                    live_members = live._select_unconflicted_candidates(
                        {"candidates": candidates}, set(), set(), inputs["top"]
                    )
                    self.assertEqual(live_members, selected.sample_id.tolist())
                    live_values = {symbol: (score, count) for symbol, score, count in live_ranking}
                    for row in selected.itertuples():
                        self.assertAlmostEqual(row.liquidity_score, live_values[row.sample_id][0])
                        column = int(np.flatnonzero(symbols == row.sample_id)[0])
                        self.assertEqual(live_values[row.sample_id][1], completed[index, column])

    def test_exact_cutoff_ties_are_lexical_regardless_of_input_order(self):
        for names in itertools.permutations(["C", "B", "A", "D"]):
            symbols = np.asarray(names)
            for issuers in (None, {name: name for name in names}):
                indices = ranking.top_ranked_indices(np.ones(4), symbols, 2, issuers=issuers)
                self.assertEqual(symbols[indices].tolist(), ["A", "B"])

    def test_blocked_best_share_class_does_not_exclude_the_other_class(self):
        symbols = np.asarray(["GOOGL", "GOOG", "NVDA", "AAPL"])
        issuers = {"GOOGL": "Alphabet", "GOOG": "Alphabet"}
        selected = ranking.top_ranked_indices(
            np.asarray([4., 3., 2., 1.]), symbols, 2, issuers=issuers,
            eligible_mask=np.asarray([False, True, True, True]),
        )
        self.assertEqual(symbols[selected].tolist(), ["GOOG", "NVDA"])

    def test_shared_scores_and_counts_ignore_same_day_and_future_data(self):
        inputs = ranking_inputs()
        baseline = inputs["dollar_volume"]
        revised = baseline.copy()
        revised[15:] *= 10_000
        for scheme in ("dollar_ema", "turnover_stability"):
            np.testing.assert_equal(
                ranking.liquidity_scores(baseline, scheme, 3, 3)[:16],
                ranking.liquidity_scores(revised, scheme, 3, 3)[:16],
            )
        revised[15:] = np.nan
        np.testing.assert_equal(
            ranking.causal_completed_trading_days(baseline)[:16],
            ranking.causal_completed_trading_days(revised)[:16],
        )

    def test_duplicate_live_bars_cannot_inflate_minimum_history(self):
        dates = pd.date_range("2026-06-01", periods=3, freq="B")
        repeated = {"t": "2026-06-01T04:00:00Z", "v": 100., "vw": 1.}
        with self.assertRaisesRegex(ValueError, "duplicate completed session"):
            live.completed_liquidity_ranking(
                {"AAPL": [repeated, repeated]}, [day.date() for day in dates],
                dates[-1].date(), 2, 1, 2,
            )

    def test_issuer_identity_is_shared_and_unknown_symbols_remain_distinct(self):
        master = {
            "GOOG": {"name": "Alphabet Inc. - Class C Capital Stock"},
            "GOOGL": {"name": "Alphabet Inc. - Class A Common Stock"},
        }
        issuers = ranking.build_issuer_map(["GOOG", "GOOGL", "UNKNOWN", "OTHER"], master)
        self.assertEqual(issuers["GOOG"], issuers["GOOGL"])
        self.assertNotEqual(issuers["UNKNOWN"], issuers["OTHER"])

    def test_mismatched_constraints_fail_instead_of_broadcasting(self):
        for kwargs in ({"completed_days": np.ones((2, 1))}, {"eligible_mask": np.ones((2, 1))}):
            with self.assertRaises(ValueError):
                ranking.top_ranked_indices(np.ones(2), np.asarray(["A", "B"]), 1, **kwargs)


if __name__ == "__main__":
    unittest.main()
