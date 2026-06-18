from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataPrepResult:
    current_period: pd.DataFrame
    previous_period: pd.DataFrame
    baseline_period: pd.DataFrame
    full_dataset: pd.DataFrame
    data_quality: dict[str, int]
    processing_trace: list[str]
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    previous_start: pd.Timestamp
    previous_end: pd.Timestamp
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp


class DataPrepTool:
    """Load, clean, join, and split the three demo source tables."""

    def __init__(self, data_dir: str | Path = "data/ymyc_anomaly_demo") -> None:
        self.data_dir = Path(data_dir)

    def run(self, window_days: int = 30) -> DataPrepResult:
        if window_days < 1:
            raise ValueError("window_days must be at least 1")

        users = pd.read_csv(self.data_dir / "users.csv")
        content = pd.read_csv(self.data_dir / "content.csv")
        events = pd.read_csv(self.data_dir / "events.csv")
        trace = [
            f"Loaded {len(users):,} user rows",
            f"Loaded {len(content):,} content rows",
            f"Loaded {len(events):,} event rows",
        ]

        users["signup_date"] = pd.to_datetime(users["signup_date"], errors="coerce")
        content["publish_time"] = pd.to_datetime(content["publish_time"], errors="coerce")
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
        events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
        trace.append("Standardized signup, publish, and event date columns")

        duplicate_event_count = int(events["event_id"].duplicated().sum())
        events = events.drop_duplicates(subset="event_id", keep="first").copy()
        trace.append(f"Removed {duplicate_event_count:,} duplicate event rows")

        missing_completion_count = int(events["completion_flag"].isna().sum())
        events["completion_flag"] = pd.to_numeric(events["completion_flag"], errors="coerce")
        events["watch_time_seconds"] = pd.to_numeric(events["watch_time_seconds"], errors="coerce").fillna(0)
        events["click_flag"] = pd.to_numeric(events["click_flag"], errors="coerce").fillna(0).astype(int)

        users["user_tier"] = users["user_tier"].fillna("unknown")
        users["investor_interest_type"] = users["investor_interest_type"].fillna("unknown")
        users["country"] = users["country"].fillna("unknown")
        content["macro_topic"] = content["macro_topic"].fillna("unknown")
        content["content_format"] = content["content_format"].fillna("unknown")
        content["source_type"] = content["source_type"].fillna("unknown")
        trace.append("Filled analysis-safe defaults for missing dimensions and numeric event fields")

        joined = events.merge(users, on="user_id", how="left", validate="many_to_one")
        joined = joined.merge(content, on="content_id", how="left", validate="many_to_one")
        trace.append("Joined events to user profiles and content metadata")

        missing_user_matches = int(joined["user_tier"].isna().sum())
        missing_content_matches = int(joined["macro_topic"].isna().sum())
        joined["user_tier"] = joined["user_tier"].fillna("unknown")
        joined["investor_interest_type"] = joined["investor_interest_type"].fillna("unknown")
        joined["country"] = joined["country"].fillna("unknown")
        joined["macro_topic"] = joined["macro_topic"].fillna("unknown")
        joined["content_format"] = joined["content_format"].fillna("unknown")
        joined["source_type"] = joined["source_type"].fillna("unknown")
        joined["surface"] = joined["surface"].fillna("unknown")
        joined["is_trending_flag"] = pd.to_numeric(
            joined["is_trending_flag"], errors="coerce"
        ).fillna(0).astype(int)

        dated_events = joined.dropna(subset=["event_date"]).copy()
        if dated_events.empty:
            raise ValueError("No valid event dates are available after cleaning")

        current_end = dated_events["event_date"].max().normalize()
        current_start = current_end - pd.Timedelta(days=window_days - 1)
        previous_end = current_start - pd.Timedelta(days=1)
        previous_start = previous_end - pd.Timedelta(days=window_days - 1)
        baseline_end = previous_start - pd.Timedelta(days=1)
        baseline_start = baseline_end - pd.Timedelta(days=window_days - 1)

        current = dated_events.loc[
            dated_events["event_date"].between(current_start, current_end, inclusive="both")
        ].copy()
        previous = dated_events.loc[
            dated_events["event_date"].between(previous_start, previous_end, inclusive="both")
        ].copy()
        baseline = dated_events.loc[
            dated_events["event_date"].between(baseline_start, baseline_end, inclusive="both")
        ].copy()
        trace.append(
            f"Created current window {current_start.date()} to {current_end.date()} "
            f"and previous window {previous_start.date()} to {previous_end.date()}"
        )
        trace.append(
            f"Created baseline window {baseline_start.date()} to {baseline_end.date()} "
            "for retention trend comparison"
        )

        data_quality = {
            "raw_event_rows": len(events) + duplicate_event_count,
            "event_rows_after_dedup": len(events),
            "duplicate_event_rows_removed": duplicate_event_count,
            "missing_completion_flags": missing_completion_count,
            "missing_user_matches": missing_user_matches,
            "missing_content_matches": missing_content_matches,
            "joined_rows": len(joined),
            "current_period_rows": len(current),
            "previous_period_rows": len(previous),
            "baseline_period_rows": len(baseline),
        }

        return DataPrepResult(
            current_period=current.reset_index(drop=True),
            previous_period=previous.reset_index(drop=True),
            baseline_period=baseline.reset_index(drop=True),
            full_dataset=joined.sort_values("event_date", ascending=False).reset_index(drop=True),
            data_quality=data_quality,
            processing_trace=trace,
            current_start=current_start,
            current_end=current_end,
            previous_start=previous_start,
            previous_end=previous_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
        )
