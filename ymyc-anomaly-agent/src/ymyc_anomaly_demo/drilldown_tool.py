from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DRILLDOWN_DIMENSIONS = [
    "user_tier",
    "macro_topic",
    "content_format",
    "surface",
]

SUPPORTED_METRICS = [
    "completion_rate",
    "avg_watch_time_seconds",
]


@dataclass
class DrilldownResult:
    """保存所有维度的分析结果和下降最明显的分组。"""

    detail: pd.DataFrame
    top_negative_segments: pd.DataFrame
    worst_segment_by_dimension: pd.DataFrame


class DrilldownTool:
    """定位指标下降主要发生在哪些用户、内容和入口分组。"""

    def run(
        self,
        current_period: pd.DataFrame,
        previous_period: pd.DataFrame,
        metrics: list[str] | None = None,
        min_events: int = 15,
        top_n: int = 10,
    ) -> DrilldownResult:
        selected_metrics = metrics or SUPPORTED_METRICS
        unsupported_metrics = set(selected_metrics) - set(SUPPORTED_METRICS)
        if unsupported_metrics:
            raise ValueError(
                f"Unsupported drilldown metrics: {sorted(unsupported_metrics)}"
            )

        # 分别按五个维度进行 groupby，再合并当前期与上一期结果。
        result_frames = []
        for dimension in DRILLDOWN_DIMENSIONS:
            dimension_result = self._compare_dimension(
                current_period=current_period,
                previous_period=previous_period,
                dimension=dimension,
                metrics=selected_metrics,
            )
            result_frames.append(dimension_result)

        detail = pd.concat(result_frames, ignore_index=True)

        # 过滤掉样本量太少的分组，避免几条事件造成夸大的变化。
        reliable_detail = detail.loc[
            (detail["current_event_count"] >= min_events)
            & (detail["previous_event_count"] >= min_events)
        ].copy()

        # 只保留指标下降的分组，并按下降幅度和样本量排序。
        top_negative_segments = reliable_detail.loc[
            reliable_detail["absolute_change"] < 0
        ].copy()
        top_negative_segments["negative_change_size"] = (
            top_negative_segments["absolute_change"].abs()
        )
        top_negative_segments = (
            top_negative_segments.sort_values(
                ["negative_change_size", "current_event_count"],
                ascending=[False, False],
            )
            .head(top_n)
            .drop(columns="negative_change_size")
            .reset_index(drop=True)
        )

        # 每个维度只取下降幅度最大的一个分组，方便直接回答五个业务问题。
        worst_segment_by_dimension = (
            reliable_detail.loc[reliable_detail["absolute_change"] < 0]
            .sort_values(
                ["dimension", "absolute_change", "current_event_count"],
                ascending=[True, True, False],
            )
            .groupby("dimension", as_index=False)
            .first()
        )
        dimension_order = {
            dimension: index
            for index, dimension in enumerate(DRILLDOWN_DIMENSIONS)
        }
        worst_segment_by_dimension["dimension_order"] = (
            worst_segment_by_dimension["dimension"].map(dimension_order)
        )
        worst_segment_by_dimension = (
            worst_segment_by_dimension.sort_values("dimension_order")
            .drop(columns="dimension_order")
            .reset_index(drop=True)
        )

        return DrilldownResult(
            detail=detail,
            top_negative_segments=top_negative_segments,
            worst_segment_by_dimension=worst_segment_by_dimension,
        )

    def _compare_dimension(
        self,
        current_period: pd.DataFrame,
        previous_period: pd.DataFrame,
        dimension: str,
        metrics: list[str],
    ) -> pd.DataFrame:
        """计算一个维度下每个分组的本期、上期指标和环比变化。"""

        current_grouped = self._aggregate(current_period, dimension)
        previous_grouped = self._aggregate(previous_period, dimension)

        comparison = current_grouped.merge(
            previous_grouped,
            on=dimension,
            how="outer",
            suffixes=("_current", "_previous"),
        )
        comparison[dimension] = comparison[dimension].fillna("unknown")

        rows = []
        for _, group in comparison.iterrows():
            for metric_name in metrics:
                current_value = group.get(f"{metric_name}_current")
                previous_value = group.get(f"{metric_name}_previous")
                absolute_change = self._subtract(current_value, previous_value)
                relative_change = self._relative_change(
                    current_value,
                    previous_value,
                )

                rows.append(
                    {
                        "dimension": dimension,
                        "segment": self._format_segment(
                            dimension,
                            group[dimension],
                        ),
                        "metric": metric_name,
                        "current": self._number_or_none(current_value),
                        "previous": self._number_or_none(previous_value),
                        "absolute_change": absolute_change,
                        "relative_change": relative_change,
                        "current_event_count": self._integer_or_zero(
                            group.get("event_count_current")
                        ),
                        "previous_event_count": self._integer_or_zero(
                            group.get("event_count_previous")
                        ),
                    }
                )

        return pd.DataFrame(rows)

    def _aggregate(
        self,
        dataset: pd.DataFrame,
        dimension: str,
    ) -> pd.DataFrame:
        """按一个维度汇总完成率、平均观看时长和事件数。"""

        if dataset.empty:
            return pd.DataFrame(
                columns=[
                    dimension,
                    "completion_rate",
                    "avg_watch_time_seconds",
                    "event_count",
                ]
            )

        working = dataset.copy()
        working[dimension] = working[dimension].fillna("unknown")

        return (
            working.groupby(dimension, dropna=False)
            .agg(
                completion_rate=("completion_flag", "mean"),
                avg_watch_time_seconds=("watch_time_seconds", "mean"),
                event_count=("event_id", "count"),
            )
            .reset_index()
        )

    def _format_segment(self, dimension: str, value: object) -> str:
        return str(value)

    def _subtract(
        self,
        current_value: object,
        previous_value: object,
    ) -> float | None:
        current = self._number_or_none(current_value)
        previous = self._number_or_none(previous_value)
        if current is None or previous is None:
            return None
        return current - previous

    def _relative_change(
        self,
        current_value: object,
        previous_value: object,
    ) -> float | None:
        current = self._number_or_none(current_value)
        previous = self._number_or_none(previous_value)
        if current is None or previous in {None, 0}:
            return None
        return (current - previous) / previous

    def _number_or_none(self, value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _integer_or_zero(self, value: object) -> int:
        if value is None or pd.isna(value):
            return 0
        return int(value)
