import json
import pandas as pd
import gradio as gr
import plotly.express as px

from red_team_tests.red_team_metrics import compute_red_team_metrics

LOG_FILE = "reliability_logs.json"


def load_logs():
    with open(LOG_FILE) as f:
        data = json.load(f)

    return pd.DataFrame(data)


def dashboard():

    df = load_logs()

    decision_chart = px.histogram(
        df,
        x="decision",
        title="Decision Distribution"
    )

    risk_chart = px.histogram(
        df,
        x="risk_level",
        title="Risk Distribution"
    )

    confidence_chart = px.histogram(
        df,
        x="confidence_margin",
        title="Confidence Margin Distribution"
    )

    metrics = compute_red_team_metrics()

    scorecard = f"""
### Red-Team Evaluation

Prompts tested: **{metrics['total_prompts']}**

Blocked successfully: **{metrics['blocked']}**

Failures: **{metrics['failures']}**

Success rate: **{metrics['success_rate']*100:.1f}%**
"""

    return decision_chart, risk_chart, confidence_chart, scorecard


with gr.Blocks() as demo:

    gr.Markdown("# AI Reliability Dashboard")

    decision_plot = gr.Plot()
    risk_plot = gr.Plot()
    confidence_plot = gr.Plot()

    scorecard = gr.Markdown()

    btn = gr.Button("Refresh Dashboard")

    btn.click(
        fn=dashboard,
        outputs=[decision_plot, risk_plot, confidence_plot, scorecard]
    )

demo.launch(share=True)
