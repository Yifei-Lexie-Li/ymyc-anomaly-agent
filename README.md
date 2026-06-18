# YMYC AI Content Anomaly Analysis

An LLM-powered analytics workflow for detecting performance anomalies across
macro news, video, and deep-dive content.

The application prepares behavioral data, compares performance across time
periods, identifies the most affected business segments, retrieves relevant
analysis playbooks with OpenAI Embeddings, and generates an evidence-grounded
report.

## Demo Capabilities

- Cleans and joins user, content, and engagement-event data
- Compares current, previous, and baseline analysis periods
- Calculates active users, completion rate, watch time, CTR, and retention
- Detects material KPI declines with deterministic thresholds
- Identifies the largest decline by:
  - user segment
  - macro topic
  - content format
  - distribution channel
- Retrieves relevant business guidance with OpenAI Embeddings
- Generates an LLM report that separates observed findings from hypotheses
- Presents results in an interactive Streamlit dashboard

## Architecture

```text
users.csv + content.csv + events.csv
                |
                v
         DataPrepTool
     clean, validate, and join
                |
                v
       AnomalyCheckTool
   KPI comparison and retention
                |
                v
         DrilldownTool
 identify affected business segments
                |
                v
    OpenAI Embeddings Retrieval
 retrieve definitions and playbooks
                |
                v
          ReportTool
 generate an evidence-grounded report
                |
                v
       Streamlit Dashboard
```

The application is implemented as a deterministic agentic workflow. Python and
Pandas calculate all metrics, while the LLM is used only for explanation and
report writing.

## Example Findings

The synthetic dataset currently produces an example completion-rate anomaly:

```text
Overall completion rate: 55.9% -> 49.3% (-6.6 percentage points)
Active-user retention:   72.9% -> 68.6% (-4.3 percentage points)
```

The drilldown highlights the largest decline within each business dimension:

| Business dimension | Affected segment |
| --- | --- |
| User segment | Returning users |
| Macro topic | China trade |
| Content format | Deep dive |
| Distribution channel | Newsletter |

These findings are calculated independently. The workflow does not assume they
represent the same users or events.

## Synthetic Data

The repository contains only generated demonstration data:

| Table | Description |
| --- | --- |
| `users.csv` | User tier, location, signup date, and investment interest |
| `content.csv` | Topic, format, source, sector, and publication metadata |
| `events.csv` | Clicks, watch time, completion, surface, and event timestamps |

Dataset size:

```text
180 users
60 content items
1,000 engagement events
3 analysis periods
```

No real customer or company data is included.

## RAG Knowledge Base

The local knowledge base contains:

```text
metric_definitions.md
user_engagement_playbook.md
content_performance_playbook.md
distribution_channel_playbook.md
```

The workflow:

1. Converts each Markdown section into a knowledge chunk
2. Generates embeddings with `text-embedding-3-small`
3. Embeds the anomaly and drilldown query
4. Calculates cosine similarity
5. Returns the most relevant evidence to the report model

## Project Structure

```text
ymyc-anomaly-agent/
├── app_ymyc_anomaly_demo.py
├── requirements.txt
├── render.yaml
├── .streamlit/
│   └── config.toml
├── data/
│   └── ymyc_anomaly_demo/
│       ├── users.csv
│       ├── content.csv
│       └── events.csv
├── knowledge/
│   └── ymyc_anomaly/
│       ├── metric_definitions.md
│       ├── user_engagement_playbook.md
│       ├── content_performance_playbook.md
│       └── distribution_channel_playbook.md
└── src/
    └── ymyc_anomaly_demo/
        ├── data_prep_tool.py
        ├── anomaly_check_tool.py
        ├── drilldown_tool.py
        ├── rag_retriever.py
        ├── report_tool.py
        └── workflow.py
```

## Local Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Configure the OpenAI API in the current terminal:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_REPORT_MODEL="gpt-5.5"
```

Run the dashboard:

```bash
.venv/bin/streamlit run app_ymyc_anomaly_demo.py
```

Open:

```text
http://localhost:8501
```

## Render Deployment

The repository includes a Render Blueprint in `render.yaml`.

1. Push this project to GitHub.
2. Sign in to [Render](https://dashboard.render.com/) with GitHub.
3. Select **New -> Blueprint**.
4. Choose this repository.
5. Enter `OPENAI_API_KEY` when requested.
6. Deploy the service.

Render installs dependencies with:

```bash
pip install -r requirements.txt
```

It starts the application with:

```bash
streamlit run app_ymyc_anomaly_demo.py \
  --server.port $PORT \
  --server.address 0.0.0.0
```

## Safety and Reliability

- All KPI calculations are deterministic and performed in Python
- The LLM does not calculate or modify metrics
- RAG evidence is shown alongside the generated report
- Segment findings are not presented as proven causal relationships
- The report distinguishes observed facts from possible explanations
- API keys are read from environment variables and excluded from Git

## Technology

- Python
- Pandas
- Streamlit
- Plotly
- scikit-learn
- OpenAI Embeddings
- OpenAI Responses API
- Render
