from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ymyc_anomaly_demo import AnomalyAnalysisWorkflow


st.set_page_config(
    page_title="YMYC AI Content Anomaly Monitor",
    page_icon="Y",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #e4eee8;
            --muted: #9db0a7;
            --line: #344741;
            --panel: #1a2724;
            --panel-strong: #21322e;
            --canvas: #101816;
            --green: #61b89f;
            --red: #ef786f;
            --amber: #e2aa5e;
            --blue: #79a9d8;
        }
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }
        html, body, [class*="css"] {
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background: #14201d;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--ink);
        }
        .brand-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .brand-mark {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            background: var(--green);
            color: #10201b;
            font-size: 1rem;
            font-weight: 800;
            border-radius: 6px;
        }
        .eyebrow {
            color: var(--green);
            font-size: 0.74rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .page-subtitle {
            color: var(--muted);
            margin-top: -0.5rem;
            margin-bottom: 1.1rem;
        }
        .status-strip {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0.2rem 0 1rem;
        }
        .status-chip {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 650;
        }
        .alert-panel {
            border-left: 4px solid var(--red);
            background: #312321;
            padding: 0.8rem 1rem;
            margin: 0.35rem 0 0.8rem;
        }
        .alert-title {
            color: var(--red);
            font-size: 0.78rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .alert-copy {
            margin-top: 0.2rem;
            color: var(--ink);
            font-weight: 600;
        }
        .evidence-block {
            border-left: 3px solid var(--blue);
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.75rem;
            background: #182530;
            color: var(--ink);
        }
        .evidence-meta {
            color: var(--blue);
            font-size: 0.76rem;
            font-weight: 700;
        }
        .report-shell {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            padding: 1.15rem 1.35rem;
            color: var(--ink);
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            padding: 0.8rem 0.9rem;
            min-height: 118px;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }
        div[data-testid="stMetricDelta"] {
            background: #263934;
            border-radius: 999px;
            display: inline-flex;
            padding: 0.18rem 0.45rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid var(--line);
        }
        [data-testid="stTabs"] button[role="tab"] {
            color: var(--muted) !important;
            background: transparent !important;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--green) !important;
            border-bottom-color: var(--green) !important;
        }
        [data-testid="stTabs"] button[role="tab"] p {
            color: inherit !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--ink);
        }
        [data-testid="stAlert"] {
            background: #1b2b31;
            color: var(--ink);
            border-color: #34515c;
        }
        .stSelectbox [data-baseweb="select"] > div,
        .stButton > button {
            background: var(--panel-strong);
            border-color: var(--line);
            color: var(--ink);
        }
        .stButton > button[kind="primary"] {
            background: var(--green);
            color: #10201b;
            border-color: var(--green);
            font-weight: 750;
        }
        .overview-section {
            border-top: 1px solid var(--line);
            margin-top: 1rem;
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 1.5rem;
            max-width: 1440px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_metric(metric_name: str, value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if metric_name in {"completion_rate", "ctr", "retention_rate"}:
        return f"{float(value):.1%}"
    if metric_name == "avg_watch_time_seconds":
        return f"{float(value):.1f}s"
    return f"{int(value):,}"


def format_delta(metric_name: str, value: float | int | None) -> str | None:
    if value is None:
        return None
    if metric_name in {"completion_rate", "ctr", "retention_rate"}:
        return f"{float(value):+.1%} pp"
    if metric_name == "avg_watch_time_seconds":
        return f"{float(value):+.1f}s"
    return f"{float(value):+.1f}"


def comparison_lookup(comparisons: pd.DataFrame) -> dict[str, dict]:
    return {
        str(row["metric"]): row.to_dict()
        for _, row in comparisons.iterrows()
    }


def render_metric_cards(result) -> None:
    comparisons = comparison_lookup(result.comparisons)
    metrics = [
        ("active_users", "Active Users"),
        ("completion_rate", "Completion Rate"),
        ("avg_watch_time_seconds", "Avg Watch Time"),
        ("ctr", "CTR Proxy"),
        ("retention_rate", "Retention"),
    ]
    columns = st.columns(5)

    for column, (metric_name, label) in zip(columns, metrics):
        row = comparisons.get(metric_name, {})
        column.metric(
            label,
            format_metric(metric_name, row.get("current")),
            format_delta(metric_name, row.get("absolute_change")),
            border=False,
        )


def render_anomaly_summary(result) -> None:
    if result.anomalies.empty:
        st.success("No threshold-based anomalies were detected.")
        return

    for _, anomaly in result.anomalies.iterrows():
        st.markdown(
            f"""
            <div class="alert-panel">
                <div class="alert-title">{str(anomaly['severity']).upper()} ANOMALY</div>
                <div class="alert-copy">{anomaly['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_comparison_chart(result) -> None:
    comparison = result.comparisons.copy()
    rate_metrics = comparison.loc[
        comparison["metric"].isin(
            ["completion_rate", "ctr", "retention_rate"]
        )
    ].copy()
    labels = {
        "completion_rate": "Completion",
        "ctr": "CTR",
        "retention_rate": "Retention",
    }
    rate_metrics["label"] = rate_metrics["metric"].map(labels)

    figure = go.Figure()
    figure.add_bar(
        name="Previous",
        x=rate_metrics["label"],
        y=rate_metrics["previous"] * 100,
        marker_color="#9aa6af",
    )
    figure.add_bar(
        name="Current",
        x=rate_metrics["label"],
        y=rate_metrics["current"] * 100,
        marker_color="#1f6f5f",
    )
    figure.update_layout(
        barmode="group",
        height=310,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Rate (%)",
        font_color="#dbe8e1",
        xaxis=dict(gridcolor="#2d403a", linecolor="#52655e"),
        yaxis=dict(
            title="Rate (%)",
            gridcolor="#2d403a",
            linecolor="#52655e",
        ),
        legend_orientation="h",
        legend_y=1.1,
    )
    st.plotly_chart(figure, use_container_width=True)


def render_drilldown(result) -> None:
    labels = {
        "user_tier": "User Segment",
        "macro_topic": "Content Topic",
        "content_format": "Content Format",
        "surface": "Distribution Channel",
    }
    drilldown = result.worst_segments.copy()
    drilldown["Business Dimension"] = drilldown["dimension"].map(labels)
    drilldown["Worst Segment"] = drilldown["segment"].astype(str).str.replace(
        "_", " ", regex=False
    ).str.title()
    drilldown["Current"] = drilldown.apply(
        lambda row: format_metric(row["metric"], row["current"]),
        axis=1,
    )
    drilldown["Previous"] = drilldown.apply(
        lambda row: format_metric(row["metric"], row["previous"]),
        axis=1,
    )
    drilldown["Change"] = drilldown.apply(
        lambda row: format_delta(row["metric"], row["absolute_change"]),
        axis=1,
    )
    drilldown["Sample Size"] = drilldown.apply(
        lambda row: (
            f"{int(row['previous_event_count'])} → "
            f"{int(row['current_event_count'])}"
        ),
        axis=1,
    )

    st.dataframe(
        drilldown[
            [
                "Business Dimension",
                "Worst Segment",
                "Previous",
                "Current",
                "Change",
                "Sample Size",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_evidence(result) -> None:
    st.caption(f"Retrieval query: {result.rag_query}")
    for item in result.evidence:
        st.markdown(
            f"""
            <div class="evidence-block">
                <div class="evidence-meta">
                    {item.section} · {item.source} · score {item.relevance_score}
                </div>
                <div>{item.text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_data_quality(result) -> None:
    quality = pd.DataFrame(
        [
            {
                "Check": key.replace("_", " ").title(),
                "Value": value,
            }
            for key, value in result.data_quality.items()
        ]
    )
    left, right = st.columns([1, 1.3])
    with left:
        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Processing Trace")
        for index, step in enumerate(
            result.data_prep.processing_trace,
            start=1,
        ):
            st.write(f"{index}. {step}")


def main() -> None:
    inject_styles()

    with st.sidebar:
        st.markdown("## YMYC AI")
        st.caption("Content Intelligence")
        st.divider()
        window_days = st.selectbox(
            "Analysis window",
            options=[30, 14, 7],
            index=0,
            format_func=lambda value: f"{value} days",
        )
        min_events = st.slider(
            "Minimum segment events",
            min_value=5,
            max_value=40,
            value=15,
            step=5,
        )
        evidence_count = st.slider(
            "RAG evidence count",
            min_value=1,
            max_value=5,
            value=3,
        )
        run_clicked = st.button(
            "Run anomaly analysis",
            type="primary",
            use_container_width=True,
        )
        st.divider()
        api_status = "Connected" if os.getenv("OPENAI_API_KEY") else "Not configured"
        st.caption(f"OpenAI: {api_status}")
        st.caption("Embeddings: text-embedding-3-small")

    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">Y</div>
            <div>
                <div class="eyebrow">Content Intelligence</div>
                <h1 style="margin: 0;">Anomaly Analysis Monitor</h1>
            </div>
        </div>
        <div class="page-subtitle">
            Detect content-performance shifts, isolate affected business segments,
            retrieve relevant playbooks, and generate an evidence-grounded report.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="status-strip">
            <span class="status-chip">1,000 synthetic events</span>
            <span class="status-chip">3 analysis periods</span>
            <span class="status-chip">OpenAI Embeddings RAG</span>
            <span class="status-chip">LLM report generation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not run_clicked:
        st.info(
            "Choose the analysis settings in the sidebar, then run the workflow."
        )
        return

    try:
        with st.spinner(
            "Preparing data, checking anomalies, retrieving evidence, "
            "and generating the report..."
        ):
            result = AnomalyAnalysisWorkflow().run(
                window_days=window_days,
                min_segment_events=min_events,
                evidence_count=evidence_count,
            )
    except Exception as error:
        st.error("The workflow could not complete.")
        st.exception(error)
        return

    render_metric_cards(result)
    render_anomaly_summary(result)

    overview_tab, drilldown_tab, evidence_tab, report_tab, quality_tab = st.tabs(
        [
            "Overview",
            "Business Drilldown",
            f"RAG Evidence ({len(result.evidence)})",
            "LLM Report ✓",
            "Data Quality",
        ]
    )

    with overview_tab:
        left, right = st.columns([1.35, 1])
        with left:
            st.subheader("Period Comparison")
            render_comparison_chart(result)
        with right:
            st.subheader("Retention Detail")
            retention = result.retention
            st.metric(
                "Retained Users",
                f"{int(retention['retained_users']):,}",
            )
            st.metric(
                "Lost Users",
                f"{int(retention['lost_users']):,}",
            )
            st.caption(
                f"Previous active users: "
                f"{int(retention['previous_active_users']):,}"
            )

        st.markdown(
            '<div class="overview-section"></div>',
            unsafe_allow_html=True,
        )
        evidence_col, report_col = st.columns([0.9, 1.4])
        with evidence_col:
            st.subheader("RAG Evidence")
            st.caption(
                f"{len(result.evidence)} sections retrieved with "
                "OpenAI Embeddings"
            )
            for item in result.evidence:
                st.markdown(
                    f"**{item.section}**  \n"
                    f"{item.source} · score {item.relevance_score}"
                )
        with report_col:
            st.subheader("LLM Report Preview")
            report_preview = result.report.report[:900]
            if len(result.report.report) > 900:
                report_preview += "..."
            st.markdown(
                f'<div class="report-shell">{report_preview}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Open the LLM Report tab to read the full report.")

    with drilldown_tab:
        st.subheader("Largest Decline by Business Dimension")
        st.caption(
            "Each row is calculated independently. It does not imply that "
            "the same events belong to every listed segment."
        )
        render_drilldown(result)

    with evidence_tab:
        st.subheader("Retrieved Business Evidence")
        st.caption(
            "OpenAI Embeddings selected these knowledge-base sections for "
            "the detected anomaly."
        )
        render_evidence(result)

    with report_tab:
        st.subheader("Evidence-Grounded Anomaly Report")
        st.caption(f"Generated with {result.report.model}")
        st.markdown('<div class="report-shell">', unsafe_allow_html=True)
        st.markdown(result.report.report)
        st.markdown("</div>", unsafe_allow_html=True)

    with quality_tab:
        st.subheader("Data Preparation and Quality")
        render_data_quality(result)


if __name__ == "__main__":
    main()
