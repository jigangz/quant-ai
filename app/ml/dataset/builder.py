"""
DatasetBuilder

Builds training datasets from raw market data with:
- Multi-ticker support
- Configurable date ranges
- Feature group selection
- Proper time-series splitting (no data leakage)
"""

import logging

import pandas as pd

from app.providers import get_market_provider
from app.ml.features.technical import add_technical_features
from app.ml.features.registry import feature_registry
from app.ml.dataset.schemas import (
    DatasetConfig,
    DatasetResult,
    DatasetOutput,
    TickerDataset,
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds ML-ready datasets from market data.

    Features:
    - Multi-ticker support with consistent schema
    - Time-series aware splitting (no shuffle, no leakage)
    - Configurable feature groups
    - Configurable label generation

    Usage:
        config = DatasetConfig(
            tickers=["AAPL", "MSFT"],
            feature_groups=["ta_basic", "volatility"],
            label_config=LabelConfig(horizon_days=5),
        )
        builder = DatasetBuilder(config)
        output = builder.build()

        # Use output.X_train, output.y_train, etc.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.market_provider = get_market_provider()

    def build(self) -> DatasetOutput:
        """
        Build the complete dataset.

        Returns:
            DatasetOutput with X_train, y_train, X_val, y_val, X_test, y_test, metadata
        """
        logger.info(f"Building dataset for {len(self.config.tickers)} tickers")

        # 1. Fetch and process data for each ticker
        all_data = []
        ticker_stats = []
        tickers_processed = []
        tickers_skipped = []

        for ticker in self.config.tickers:
            try:
                df = self._process_ticker(ticker)

                if len(df) < self.config.min_samples_per_ticker:
                    logger.warning(
                        f"Skipping {ticker}: only {len(df)} samples "
                        f"(min: {self.config.min_samples_per_ticker})"
                    )
                    tickers_skipped.append(ticker)
                    continue

                all_data.append(df)
                tickers_processed.append(ticker)

                # Collect stats
                ticker_stats.append(
                    TickerDataset(
                        ticker=ticker,
                        n_samples=len(df),
                        n_features=len(
                            [
                                c
                                for c in df.columns
                                if c not in ["ticker", "date", "label"]
                            ]
                        ),
                        date_range=(
                            df["date"].min().isoformat(),
                            df["date"].max().isoformat(),
                        ),
                        label_distribution=df["label"].value_counts().to_dict(),
                    )
                )

                logger.info(f"Processed {ticker}: {len(df)} samples")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                tickers_skipped.append(ticker)

        if not all_data:
            raise ValueError("No valid data for any ticker")

        # 2. Combine all ticker data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(combined_df)} samples")

        # 3. Get feature columns
        feature_cols = self._get_feature_columns(combined_df)

        # 4. Time-series split (by date, across all tickers)
        train_df, val_df, test_df = self._time_series_split(combined_df)

        # 5. Extract X, y
        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_val = val_df[feature_cols]
        y_val = val_df["label"]
        X_test = test_df[feature_cols]
        y_test = test_df["label"]

        # 6. Build metadata
        metadata = DatasetResult(
            config=self.config,
            tickers_processed=tickers_processed,
            tickers_skipped=tickers_skipped,
            total_samples=len(combined_df),
            n_features=len(feature_cols),
            feature_names=feature_cols,
            ticker_stats=ticker_stats,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            train_date_range=(
                train_df["date"].min().isoformat(),
                train_df["date"].max().isoformat(),
            ),
            val_date_range=(
                val_df["date"].min().isoformat(),
                val_df["date"].max().isoformat(),
            ),
            test_date_range=(
                test_df["date"].min().isoformat(),
                test_df["date"].max().isoformat(),
            ),
        )

        logger.info(
            f"Dataset built: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return DatasetOutput(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
        )

    def _process_ticker(self, ticker: str) -> pd.DataFrame:
        """Process a single ticker: fetch data, add features, add labels."""

        # Fetch market data
        df = self.market_provider.fetch(
            ticker=ticker,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        if df.empty:
            raise ValueError(f"No data for {ticker}")

        # Add technical features
        df = add_technical_features(df)

        # Add labels
        df = self._add_labels(df)

        # Drop rows without labels (future not available)
        df = df.dropna(subset=["label"])

        # Optionally drop rows with NaN features
        if self.config.drop_na_features:
            feature_cols = self._get_feature_columns(df)
            df = df.dropna(subset=feature_cols)

        return df

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add labels based on config."""
        df = df.sort_values("date").copy()

        horizon = self.config.label_config.horizon_days
        threshold = self.config.label_config.threshold

        # Calculate future return
        df["future_price"] = df["close"].shift(-horizon)
        df["future_return"] = (df["future_price"] - df["close"]) / df["close"]

        # Generate label based on type
        if self.config.label_config.label_type == "direction":
            df["label"] = (df["future_return"] > threshold).astype(int)
        else:  # return
            df["label"] = df["future_return"]

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature column names based on configured feature groups."""
        # Use feature registry to get feature names
        feature_cols = feature_registry.get_feature_names(self.config.feature_groups)

        # Filter to only columns that exist in the DataFrame
        existing_cols = [col for col in feature_cols if col in df.columns]

        if len(existing_cols) < len(feature_cols):
            missing = set(feature_cols) - set(existing_cols)
            logger.warning(f"Missing features in DataFrame: {missing}")

        return existing_cols

    def _time_series_split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by date (NOT by row index).

        This ensures:
        - No future data leaks into training
        - All tickers share the same date boundaries
        - Consistent temporal ordering
        """
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Get unique dates
        unique_dates = df["date"].unique()
        n_dates = len(unique_dates)

        # Calculate split points
        train_end_idx = int(n_dates * self.config.split_config.train_ratio)
        val_end_idx = int(
            n_dates
            * (
                self.config.split_config.train_ratio
                + self.config.split_config.val_ratio
            )
        )

        train_end_date = unique_dates[train_end_idx - 1]
        val_end_date = unique_dates[val_end_idx - 1]

        # Split by date
        train_df = df[df["date"] <= train_end_date].copy()
        val_df = df[(df["date"] > train_end_date) & (df["date"] <= val_end_date)].copy()
        test_df = df[df["date"] > val_end_date].copy()

        logger.info(
            f"Split dates: train <= {train_end_date}, "
            f"val <= {val_end_date}, test > {val_end_date}"
        )

        return train_df, val_df, test_df
