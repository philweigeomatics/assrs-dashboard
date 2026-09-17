"""
strategies — the screens from the Streamlit app, as importable logic.

Each module holds the scoring or rules for one strategy, extracted from its
page so that the Streamlit app and the new SPA cannot drift into disagreeing
about what a signal means. The extraction is deliberately verbatim: same
weights, same thresholds, same verdict cuts. Where the page interleaved the
maths with st.* calls and Tushare fetches, the maths here is a pure function of
a frame plus whatever facts the frame cannot supply, so it can be tested
without a network and re-used by a nightly job.

Ported so far:
  * t_trading      — 做T候选, structural suitability for intraday round-trips
  * mean_reversion — 反转候选, sentiment-driven oversold snapbacks

Still on the Streamlit side: pair_trader (cointegration and spread z-score)
and lead_lag_analysis (cross-correlation at lag). Both are chart-led rather
than table-led and need their own visualisations, not just a port.
"""

from . import mean_reversion, t_trading

__all__ = ["mean_reversion", "t_trading"]
