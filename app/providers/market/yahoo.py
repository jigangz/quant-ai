"""
Yahoo Finance Market Data Provider

Fetches OHLCV data from Yahoo Finance using yfinance library.
"""

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app.providers.base import MarketProvider

logger = logging.getLogger(__name__)


class YahooMarketProvider(MarketProvider):
    """Yahoo Finance market data provider."""
    
    @property
    def provider_name(self) -> str:
        return "yahoo"
    
    def fetch(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: str = "1d",
        period: str | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            interval: Data interval ("1d", "1h", "5m", etc.)
            period: Alternative to date range ("1mo", "3mo", "1y", "max")
        
        Returns:
            DataFrame with columns: [ticker, date, open, high, low, close, volume]
        """
        ticker = self.validate_ticker(ticker)
        start_date, end_date = self.validate_date_range(start_date, end_date)
        
        logger.info(f"Fetching {ticker} from Yahoo Finance")
        
        try:
            # Use period if no date range specified
            if period and not (start_date or end_date):
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                )
            else:
                # Default to max if nothing specified
                if not start_date and not end_date:
                    period = "max"
                    df = yf.download(
                        ticker,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                    )
                else:
                    # Add 1 day to end_date for inclusive range
                    end_str = (end_date + timedelta(days=1)).isoformat() if end_date else None
                    start_str = start_date.isoformat() if start_date else None
                    
                    df = yf.download(
                        ticker,
                        start=start_str,
                        end=end_str,
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                    )
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
            
            # Normalize the DataFrame
            df = self._normalize_dataframe(df, ticker)
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise
    
    def _normalize_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalize Yahoo Finance DataFrame to standard schema."""
        df = df.reset_index()
        
        # Handle MultiIndex columns (Yahoo edge case)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Rename columns to standard schema
        df = df.rename(
            columns={
                "Date": "date",
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        
        # Add ticker column
        df["ticker"] = ticker.upper()
        
        # Ensure date is date type (not datetime)
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.date
        
        # Select and order columns
        return df[["ticker", "date", "open", "high", "low", "close", "volume"]]
