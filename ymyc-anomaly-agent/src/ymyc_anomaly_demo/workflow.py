from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .anomaly_check_tool import AnomalyCheckResult, AnomalyCheckTool
from .data_prep_tool import DataPrepResult, DataPrepTool
from .drilldown_tool import DrilldownResult, DrilldownTool
from .rag_retriever import RAGRetriever, RetrievedEvidence
from .report_tool import ReportResult, ReportTool


@dataclass
class WorkflowResult:
    """保存完整异常分析流程的所有输出。"""

    data_prep: DataPrepResult
    anomaly_check: AnomalyCheckResult
    drilldown: DrilldownResult
    rag_query: str
    evidence: list[RetrievedEvidence]
    report: ReportResult

    @property
    def current_metrics(self) -> dict[str, float | int | None]:
        return self.anomaly_check.current_metrics

    @property
    def previous_metrics(self) -> dict[str, float | int | None]:
        return self.anomaly_check.previous_metrics

    @property
    def comparisons(self) -> pd.DataFrame:
        return pd.DataFrame(self.anomaly_check.comparisons)

    @property
    def anomalies(self) -> pd.DataFrame:
        return pd.DataFrame(self.anomaly_check.anomalies)

    @property
    def retention(self) -> dict[str, float | int | None]:
        return self.anomaly_check.retention

    @property
    def worst_segments(self) -> pd.DataFrame:
        return self.drilldown.worst_segment_by_dimension

    @property
    def data_quality(self) -> dict[str, int]:
        return self.data_prep.data_quality


class AnomalyAnalysisWorkflow:
    """按固定顺序编排数据准备、异常检查、下钻、RAG 和 LLM 报告。"""

    def __init__(
        self,
        data_prep_tool: DataPrepTool | None = None,
        anomaly_check_tool: AnomalyCheckTool | None = None,
        drilldown_tool: DrilldownTool | None = None,
        rag_retriever: RAGRetriever | None = None,
        report_tool: ReportTool | None = None,
    ) -> None:
        # 允许从外部传入 Tool，方便以后测试或替换实现。
        self.data_prep_tool = data_prep_tool or DataPrepTool()
        self.anomaly_check_tool = anomaly_check_tool or AnomalyCheckTool()
        self.drilldown_tool = drilldown_tool or DrilldownTool()
        self.rag_retriever = rag_retriever or RAGRetriever()
        self.report_tool = report_tool or ReportTool()

    def run(
        self,
        window_days: int = 30,
        min_segment_events: int = 15,
        evidence_count: int = 3,
    ) -> WorkflowResult:
        # 第一步：读取、清洗、合并三张表，并切分三个时间周期。
        prep_result = self.data_prep_tool.run(window_days=window_days)

        # 第二步：计算 KPI、Retention、环比变化和总体异常。
        anomaly_result = self.anomaly_check_tool.run(
            current_period=prep_result.current_period,
            previous_period=prep_result.previous_period,
            baseline_period=prep_result.baseline_period,
        )

        # 第三步：有异常时优先下钻触发异常的指标。
        # 如果没有异常，则默认分析完成率和平均观看时间。
        drilldown_metrics = self._select_drilldown_metrics(anomaly_result)
        drilldown_result = self.drilldown_tool.run(
            current_period=prep_result.current_period,
            previous_period=prep_result.previous_period,
            metrics=drilldown_metrics,
            min_events=min_segment_events,
        )

        # 第四步：将异常和最差业务分组组合成 RAG 检索问题。
        rag_query = self._build_rag_query(
            anomaly_result=anomaly_result,
            drilldown_result=drilldown_result,
        )

        # 第五步：使用 OpenAI Embeddings 检索相关业务知识。
        evidence = self.rag_retriever.retrieve(
            query=rag_query,
            top_k=evidence_count,
        )

        # 第六步：将确定性分析结果和 RAG 证据交给 LLM 生成报告。
        report = self.report_tool.run(
            comparisons=anomaly_result.comparisons,
            anomalies=anomaly_result.anomalies,
            retention=anomaly_result.retention,
            drilldown_rows=drilldown_result.worst_segment_by_dimension.to_dict(
                orient="records"
            ),
            evidence=evidence,
        )

        return WorkflowResult(
            data_prep=prep_result,
            anomaly_check=anomaly_result,
            drilldown=drilldown_result,
            rag_query=rag_query,
            evidence=evidence,
            report=report,
        )

    def _select_drilldown_metrics(
        self,
        anomaly_result: AnomalyCheckResult,
    ) -> list[str] | None:
        """选择需要进一步下钻的内容表现指标。"""

        supported_metrics = {
            "completion_rate",
            "avg_watch_time_seconds",
        }
        selected_metrics = [
            str(anomaly["metric"])
            for anomaly in anomaly_result.anomalies
            if anomaly["metric"] in supported_metrics
        ]
        return selected_metrics or None

    def _build_rag_query(
        self,
        anomaly_result: AnomalyCheckResult,
        drilldown_result: DrilldownResult,
    ) -> str:
        """把数据分析结果转换成适合语义检索的自然语言问题。"""

        query_parts = [
            "Explain possible causes and recommended investigation steps "
            "for the detected content performance anomalies."
        ]

        for anomaly in anomaly_result.anomalies:
            query_parts.append(str(anomaly["message"]))

        for _, row in drilldown_result.worst_segment_by_dimension.iterrows():
            query_parts.append(
                f"The {row['dimension']} segment {row['segment']} had a "
                f"decline in {row['metric']}."
            )

        retention_change = anomaly_result.retention.get(
            "retention_rate_change"
        )
        if retention_change is not None and float(retention_change) < 0:
            query_parts.append(
                "Active user retention also declined versus the prior period."
            )

        return " ".join(query_parts)
