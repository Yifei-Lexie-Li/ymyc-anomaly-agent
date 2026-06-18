# YMYC AI Anomaly Agent

This project analyzes content performance data and detects anomalies.

It can:

- Clean and combine user, content, and event data
- Compare current and previous periods
- Detect changes in completion rate, watch time, CTR, and retention
- Find which user group, topic, content format, or channel performed worse
- Use OpenAI Embeddings to retrieve relevant business knowledge
- Use an LLM to generate an anomaly analysis report

## Workflow

Data Processing → Anomaly Detection → Drilldown → RAG → LLM Report
