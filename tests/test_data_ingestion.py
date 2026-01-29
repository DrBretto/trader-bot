"""Tests for data ingestion modules."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.ingest_prices import fetch_stooq_daily
from src.steps.ingest_fred import fetch_fred_series, forward_fill_missing, get_latest_values
from src.steps.ingest_gdelt import fetch_gdelt_daily_aggregate


class TestIngestPrices:
    """Tests for price ingestion."""

    @patch('src.steps.ingest_prices.requests.get')
    def test_fetch_stooq_daily_parses_csv(self, mock_get):
        """Test that Stooq CSV is parsed correctly."""
        # Use recent dates that will pass the lookback filter
        from datetime import datetime, timedelta
        date1 = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        date2 = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

        mock_response = Mock()
        mock_response.text = f"""Date,Open,High,Low,Close,Volume
{date1},450.23,452.10,448.50,451.80,85342100
{date2},448.00,451.00,447.50,450.20,75000000
"""
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_stooq_daily('SPY', lookback_days=30)

        assert len(result) == 2
        assert 'date' in result.columns
        assert 'symbol' in result.columns
        assert 'close' in result.columns
        assert result.iloc[0]['symbol'] == 'SPY'

    @patch('src.steps.ingest_prices.requests.get')
    def test_fetch_stooq_handles_empty_response(self, mock_get):
        """Test that empty response returns empty DataFrame."""
        mock_response = Mock()
        mock_response.text = "No data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_stooq_daily('INVALID', lookback_days=30)

        assert len(result) == 0

    @patch('src.steps.ingest_prices.requests.get')
    def test_fetch_stooq_handles_network_error(self, mock_get):
        """Test that network errors are handled gracefully."""
        mock_get.side_effect = Exception("Network error")

        result = fetch_stooq_daily('SPY', lookback_days=30)

        assert len(result) == 0


class TestIngestFred:
    """Tests for FRED ingestion."""

    @patch('src.steps.ingest_fred.requests.get')
    def test_fetch_fred_series_parses_json(self, mock_get):
        """Test that FRED JSON is parsed correctly."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'observations': [
                {'date': '2025-01-27', 'value': '4.25'},
                {'date': '2025-01-24', 'value': '4.20'},
                {'date': '2025-01-23', 'value': '.'}  # Missing value
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_fred_series('DGS10', 'test_key', lookback_days=30)

        assert len(result) == 2  # Missing value should be excluded
        assert 'date' in result.columns
        assert 'value' in result.columns
        assert 'series_id' in result.columns

    def test_forward_fill_missing(self):
        """Test that forward fill works correctly."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-20', '2025-01-22', '2025-01-24']),
            'series_id': ['DGS10'] * 3,
            'value': [4.20, 4.25, 4.30]
        })

        result = forward_fill_missing(df, max_fill_days=5)

        # Should have filled in 2025-01-21 and 2025-01-23
        assert len(result) >= 3
        dates = result['date'].tolist()
        assert pd.Timestamp('2025-01-21') in dates
        assert pd.Timestamp('2025-01-23') in dates

    def test_get_latest_values(self):
        """Test getting latest values for each series."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-24', '2025-01-27', '2025-01-24', '2025-01-27']),
            'series_id': ['DGS2', 'DGS2', 'DGS10', 'DGS10'],
            'value': [4.10, 4.15, 4.20, 4.25]
        })

        result = get_latest_values(df)

        assert result['DGS2'] == 4.15  # Latest DGS2
        assert result['DGS10'] == 4.25  # Latest DGS10


class TestIngestGdelt:
    """Tests for GDELT ingestion."""

    def test_gdelt_returns_default_on_failure(self):
        """Test that GDELT returns defaults when fetch fails."""
        # Use a date that definitely won't exist
        result = fetch_gdelt_daily_aggregate(pd.Timestamp('2099-01-01'))

        assert result['gdelt_doc_count'] == 0
        assert result['gdelt_available'] is False

    def test_gdelt_default_structure(self):
        """Test that default GDELT data has correct structure."""
        result = fetch_gdelt_daily_aggregate(pd.Timestamp('2099-01-01'))

        assert 'gdelt_doc_count' in result
        assert 'gdelt_avg_tone' in result
        assert 'gdelt_tone_std' in result
        assert 'gdelt_neg_tone_share' in result
        assert 'gdelt_available' in result
