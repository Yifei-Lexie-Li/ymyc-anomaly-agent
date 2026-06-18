from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .rag_retriever import RetrievedEvidence


@dataclass
class ReportResult:
    """保存 LLM 生成的报告以及调用的模型名称。"""

    report: str
    model: str


class ReportTool:
    """使用 LLM 将分析事实和 RAG 证据整理成业务报告。"""

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_REPORT_MODEL", "gpt-5.5")

    def run(
        self,
        comparisons: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        retention: dict[str, Any],
        drilldown_rows: list[dict[str, Any]],
        evidence: list[RetrievedEvidence],
    ) -> ReportResult:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "The LLM report cannot be generated."
            )

        from openai import OpenAI

        # 将所有确定性分析结果整理成结构化 payload。
        # LLM 只负责解释和表达，不负责重新计算指标。
        payload = {
            "period_comparisons": comparisons,
            "detected_anomalies": anomalies,
            "retention": retention,
            "worst_segment_by_dimension": drilldown_rows,
            "retrieved_evidence": [
                {
                    "source": item.source,
                    "section": item.section,
                    "text": item.text,
                    "relevance_score": item.relevance_score,
                }
                for item in evidence
            ],
        }

        instructions = """
You are a content performance analyst for a YMYC AI-style macro news and
analysis product.

Write a concise anomaly analysis report using only the supplied analytics
results and retrieved evidence.

Rules:
- Do not invent, recalculate, or modify any metric.
- Treat drilldown results as separate segment findings. Do not assume the
  segments describe the same users or events.
- Separate observed data findings from possible explanations.
- Never present a hypothesis as a proven root cause.
- Use phrases such as "possible explanation", "may indicate", and
  "recommended investigation".
- Cite evidence using its section name in square brackets.
- Keep the report business-readable and action-oriented.

Use exactly these sections:
1. Executive Summary
2. KPI and Retention Changes
3. Affected Segments
4. Possible Explanations
5. Recommended Investigations
6. Evidence Used
""".strip()

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=True, default=str),
        )

        report = (response.output_text or "").strip()
        if not report:
            raise RuntimeError("The LLM returned an empty report.")

        return ReportResult(
            report=report,
            model=self.model,
        )
