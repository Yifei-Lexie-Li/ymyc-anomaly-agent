from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AnomalyCheckResult:
    """保存 KPI 计算、环比比较和异常检查的最终结果。"""

    current_metrics: dict[str, float | int | None]
    previous_metrics: dict[str, float | int | None]
    comparisons: list[dict[str, float | int | str | None]]
    anomalies: list[dict[str, float | str]]
    retention: dict[str, float | int | None]


class AnomalyCheckTool:
    """计算两个周期的 KPI，进行环比，并检查是否存在异常。"""

    def run(
        self,
        current_period: pd.DataFrame,
        previous_period: pd.DataFrame,
        baseline_period: pd.DataFrame,
    ) -> AnomalyCheckResult:
        # 第一步：分别计算当前周期和上一周期的核心 KPI。
        current_metrics = self._calculate_metrics(current_period)
        previous_metrics = self._calculate_metrics(previous_period)

        # 第二步：计算每个 KPI 的绝对变化和相对变化。
        comparisons = self._compare_metrics(current_metrics, previous_metrics)

        # 第三步：计算用户从一个周期到下一个周期的留存情况。
        retention = self._calculate_retention(
            current_period=current_period,
            previous_period=previous_period,
            baseline_period=baseline_period,
        )
        comparisons.append(
            {
                "metric": "retention_rate",
                "current": retention["current_retention_rate"],
                "previous": retention["previous_retention_rate"],
                "absolute_change": retention["retention_rate_change"],
                "relative_change": self._relative_change(
                    retention["current_retention_rate"],
                    retention["previous_retention_rate"],
                ),
            }
        )

        # 第四步：根据业务阈值判断哪些变化需要被标记为异常。
        anomalies = self._detect_anomalies(comparisons)

        return AnomalyCheckResult(
            current_metrics=current_metrics,
            previous_metrics=previous_metrics,
            comparisons=comparisons,
            anomalies=anomalies,
            retention=retention,
        )

    def _calculate_metrics(self, dataset: pd.DataFrame) -> dict[str, float | int | None]:
        """计算一个时间周期内的四个核心指标。"""

        if dataset.empty:
            return {
                "active_users": 0,
                "completion_rate": None,
                "avg_watch_time_seconds": None,
                "ctr": None,
            }

        # errors="coerce" 会将无法识别的值转成缺失值，避免计算时报错。
        completion = pd.to_numeric(dataset["completion_flag"], errors="coerce")
        watch_time = pd.to_numeric(dataset["watch_time_seconds"], errors="coerce")
        clicks = pd.to_numeric(dataset["click_flag"], errors="coerce")

        return {
            # 去重后的 user_id 数量代表这个周期的活跃用户数。
            "active_users": int(dataset["user_id"].nunique()),
            # completion_flag 的平均值就是内容完成率。
            "completion_rate": self._mean_or_none(completion),
            # 所有事件的平均观看秒数。
            "avg_watch_time_seconds": self._mean_or_none(watch_time),
            # 当前简化版用 click_flag 的平均值作为 CTR proxy。
            "ctr": self._mean_or_none(clicks),
        }

    def _compare_metrics(
        self,
        current: dict[str, float | int | None],
        previous: dict[str, float | int | None],
    ) -> list[dict[str, float | int | str | None]]:
        """比较当前周期与上一周期，生成环比结果。"""

        comparisons = []

        for metric_name, current_value in current.items():
            previous_value = previous.get(metric_name)
            absolute_change = None
            relative_change = None

            if current_value is not None and previous_value is not None:
                # 绝对变化：当前值 - 上一期值。
                # 对完成率和 CTR 来说，这代表百分点变化。
                absolute_change = float(current_value) - float(previous_value)

                # 相对变化：(当前值 - 上一期值) / 上一期值。
                if float(previous_value) != 0:
                    relative_change = absolute_change / float(previous_value)

            comparisons.append(
                {
                    "metric": metric_name,
                    "current": current_value,
                    "previous": previous_value,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                }
            )

        return comparisons

    def _calculate_retention(
        self,
        current_period: pd.DataFrame,
        previous_period: pd.DataFrame,
        baseline_period: pd.DataFrame,
    ) -> dict[str, float | int | None]:
        """计算活动留存：上一周期活跃用户在下一周期仍然活跃的比例。"""

        current_users = set(current_period["user_id"].dropna().unique())
        previous_users = set(previous_period["user_id"].dropna().unique())
        baseline_users = set(baseline_period["user_id"].dropna().unique())

        # 当前留存：上一期活跃用户中，本期仍活跃的人数占比。
        retained_current_users = previous_users & current_users
        current_retention_rate = (
            len(retained_current_users) / len(previous_users)
            if previous_users
            else None
        )

        # 历史留存：上上期活跃用户中，上期仍活跃的人数占比。
        retained_previous_users = baseline_users & previous_users
        previous_retention_rate = (
            len(retained_previous_users) / len(baseline_users)
            if baseline_users
            else None
        )

        retention_rate_change = None
        if current_retention_rate is not None and previous_retention_rate is not None:
            retention_rate_change = current_retention_rate - previous_retention_rate

        return {
            "previous_active_users": len(previous_users),
            "retained_users": len(retained_current_users),
            "lost_users": len(previous_users - current_users),
            "current_retention_rate": current_retention_rate,
            "previous_retention_rate": previous_retention_rate,
            "retention_rate_change": retention_rate_change,
        }

    def _detect_anomalies(
        self,
        comparisons: list[dict[str, float | int | str | None]],
    ) -> list[dict[str, float | str]]:
        """使用简单业务阈值判断环比下降是否构成异常。"""

        # 每个指标使用的变化字段和报警阈值不同。
        # active_users 使用相对变化；其他指标使用绝对变化。
        thresholds = {
            "active_users": ("relative_change", -0.10),
            "completion_rate": ("absolute_change", -0.05),
            "avg_watch_time_seconds": ("absolute_change", -15.0),
            "ctr": ("absolute_change", -0.05),
            "retention_rate": ("absolute_change", -0.05),
        }
        anomalies = []

        for comparison in comparisons:
            metric_name = str(comparison["metric"])
            change_field, threshold = thresholds[metric_name]
            change_value = comparison.get(change_field)

            # 没有可比较数据，或下降幅度未超过阈值时，不产生异常。
            if change_value is None or float(change_value) > threshold:
                continue

            # 超过阈值 1.5 倍时标记为 high，否则标记为 medium。
            severity = "high" if float(change_value) <= threshold * 1.5 else "medium"
            anomalies.append(
                {
                    "metric": metric_name,
                    "severity": severity,
                    "change": float(change_value),
                    "message": self._anomaly_message(metric_name, float(change_value)),
                }
            )

        return anomalies

    def _anomaly_message(self, metric_name: str, change: float) -> str:
        if metric_name == "active_users":
            return f"Active users decreased by {abs(change):.1%} versus the previous period."
        if metric_name in {"completion_rate", "ctr", "retention_rate"}:
            label = metric_name.replace("_", " ").title()
            return f"{label} decreased by {abs(change):.1%} percentage points versus the previous period."
        return f"Average watch time decreased by {abs(change):.1f} seconds versus the previous period."

    def _relative_change(
        self,
        current_value: float | int | None,
        previous_value: float | int | None,
    ) -> float | None:
        """安全计算相对变化。"""

        if current_value is None or previous_value in {None, 0}:
            return None
        return (float(current_value) - float(previous_value)) / float(previous_value)

    def _mean_or_none(self, values: pd.Series) -> float | None:
        """安全计算平均值；没有有效数据时返回 None。"""

        valid_values = values.dropna()
        if valid_values.empty:
            return None
        return float(valid_values.mean())
