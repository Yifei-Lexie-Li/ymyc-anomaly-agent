from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/ymyc_anomaly_demo")
RANDOM_SEED = 42


def generate_demo_data(output_dir: str | Path = DATA_DIR) -> dict[str, int]:
    """Generate three realistic YMYC-style tables with exactly 1,000 events."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    users = _generate_users(rng)
    content = _generate_content(rng)
    events = _generate_events(rng, users, content)

    users.to_csv(output_path / "users.csv", index=False)
    content.to_csv(output_path / "content.csv", index=False)
    events.to_csv(output_path / "events.csv", index=False)

    return {
        "users": len(users),
        "content": len(content),
        "events": len(events),
    }


def _generate_users(rng: np.random.Generator) -> pd.DataFrame:
    user_ids = [f"u{index:03d}" for index in range(1, 181)]
    signup_dates = pd.to_datetime("2025-06-01") + pd.to_timedelta(
        rng.integers(0, 270, size=len(user_ids)),
        unit="D",
    )

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": signup_dates,
            "user_tier": rng.choice(
                ["new", "returning", "high_engagement"],
                size=len(user_ids),
                p=[0.30, 0.45, 0.25],
            ),
            "investor_interest_type": rng.choice(
                ["macro", "single_stock", "sector"],
                size=len(user_ids),
                p=[0.45, 0.30, 0.25],
            ),
            "country": rng.choice(
                ["US", "CA", "SG", "UK"],
                size=len(user_ids),
                p=[0.55, 0.15, 0.15, 0.15],
            ),
        }
    )


def _generate_content(rng: np.random.Generator) -> pd.DataFrame:
    content_ids = [f"c{index:03d}" for index in range(1, 61)]
    formats = rng.choice(
        ["article", "video", "deep_dive"],
        size=len(content_ids),
        p=[0.40, 0.35, 0.25],
    )

    return pd.DataFrame(
        {
            "content_id": content_ids,
            "macro_topic": rng.choice(
                ["inflation", "rates", "fed", "jobs", "oil", "china_trade"],
                size=len(content_ids),
            ),
            "content_format": formats,
            "sector": rng.choice(
                ["technology", "financials", "energy", "industrial"],
                size=len(content_ids),
            ),
            "publish_time": pd.to_datetime("2025-12-15")
            + pd.to_timedelta(rng.integers(0, 105, size=len(content_ids)), unit="D"),
            "is_trending_flag": rng.choice([0, 1], size=len(content_ids), p=[0.65, 0.35]),
            "source_type": rng.choice(
                ["editorial", "ai_generated"],
                size=len(content_ids),
                p=[0.60, 0.40],
            ),
        }
    )


def _generate_events(
    rng: np.random.Generator,
    users: pd.DataFrame,
    content: pd.DataFrame,
) -> pd.DataFrame:
    all_users = users["user_id"].tolist()

    # 三个周期包含留存、流失和新增用户。
    baseline_users = all_users[:110]
    previous_users = baseline_users[:88] + all_users[110:137]
    current_users = previous_users[:80] + all_users[137:180]

    periods = [
        {
            "name": "baseline",
            "start": pd.Timestamp("2026-01-01"),
            "days": 30,
            "event_count": 250,
            "active_users": baseline_users,
        },
        {
            "name": "previous",
            "start": pd.Timestamp("2026-01-31"),
            "days": 30,
            "event_count": 330,
            "active_users": previous_users,
        },
        {
            "name": "current",
            "start": pd.Timestamp("2026-03-02"),
            "days": 30,
            "event_count": 420,
            "active_users": current_users,
        },
    ]

    content_lookup = content.set_index("content_id").to_dict(orient="index")
    content_ids = content["content_id"].tolist()
    rows: list[dict[str, object]] = []

    for period in periods:
        for _ in range(period["event_count"]):
            user_id = str(rng.choice(period["active_users"]))
            content_id = str(rng.choice(content_ids))
            content_row = content_lookup[content_id]
            event_date = period["start"] + pd.Timedelta(
                days=int(rng.integers(0, period["days"]))
            )

            completion_probability = _completion_probability(
                period_name=str(period["name"]),
                content_format=str(content_row["content_format"]),
                is_trending=int(content_row["is_trending_flag"]),
            )
            completion_flag = int(rng.random() < completion_probability)

            watch_time_mean = _watch_time_mean(
                period_name=str(period["name"]),
                content_format=str(content_row["content_format"]),
                is_trending=int(content_row["is_trending_flag"]),
            )
            watch_time = max(1, int(rng.normal(watch_time_mean, 18)))

            click_probability = 0.46
            if int(content_row["is_trending_flag"]) == 1:
                click_probability += 0.10
            if str(period["name"]) == "current":
                click_probability -= 0.04
            click_flag = int(rng.random() < click_probability)

            rows.append(
                {
                    "user_id": user_id,
                    "content_id": content_id,
                    "event_date": event_date,
                    "event_type": rng.choice(
                        ["impression", "open", "video_start", "video_complete"],
                        p=[0.30, 0.25, 0.25, 0.20],
                    ),
                    "click_flag": click_flag,
                    "watch_time_seconds": watch_time,
                    "completion_flag": completion_flag,
                    "surface": rng.choice(
                        ["dashboard", "economics_home", "trending", "newsletter"],
                        p=[0.30, 0.30, 0.25, 0.15],
                    ),
                }
            )

    events = pd.DataFrame(rows)
    events.insert(0, "event_id", [f"e{index:04d}" for index in range(1, len(events) + 1)])
    events.insert(
        4,
        "event_time",
        events["event_date"]
        + pd.to_timedelta(rng.integers(0, 24, size=len(events)), unit="h")
        + pd.to_timedelta(rng.integers(0, 60, size=len(events)), unit="m"),
    )
    events.insert(
        4,
        "session_id",
        [f"s{value:04d}" for value in rng.integers(1, 401, size=len(events))],
    )

    # 少量缺失值用于展示数据质量处理。
    missing_indexes = rng.choice(events.index, size=25, replace=False)
    events.loc[missing_indexes, "completion_flag"] = np.nan

    return events.sort_values(["event_date", "event_id"]).reset_index(drop=True)


def _completion_probability(
    period_name: str,
    content_format: str,
    is_trending: int,
) -> float:
    probability = {
        "article": 0.58,
        "video": 0.54,
        "deep_dive": 0.50,
    }[content_format]

    if period_name == "previous":
        probability -= 0.02
    if period_name == "current":
        probability -= 0.08
    if period_name == "current" and content_format == "video" and is_trending == 1:
        probability -= 0.10
    return max(0.10, probability)


def _watch_time_mean(
    period_name: str,
    content_format: str,
    is_trending: int,
) -> float:
    mean = {
        "article": 52,
        "video": 78,
        "deep_dive": 96,
    }[content_format]

    if period_name == "current":
        mean -= 10
    if period_name == "current" and content_format == "video" and is_trending == 1:
        mean -= 14
    return mean


if __name__ == "__main__":
    counts = generate_demo_data()
    print(counts)
