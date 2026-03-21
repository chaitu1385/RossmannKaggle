"""AI-native features powered by Anthropic Claude.

Provides four capabilities exposed as REST API endpoints:

1. Natural Language Querying — answer planner questions about specific series
2. Anomaly Triage — rank drift alerts by business impact with suggested actions
3. Config Tuning — recommend configuration changes from backtest results
4. Forecast Commentary — generate executive summaries for S&OP meetings

All features gracefully degrade when the ``anthropic`` package is not installed
or no API key is configured.
"""
