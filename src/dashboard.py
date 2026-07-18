import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
import pandas as pd
import os

# ------------------------------------
# Load Data
# ------------------------------------
def load_data(input_dir):
    clusters = pd.read_csv(os.path.join(input_dir, "clusters.csv"))
    summary = pd.read_csv(os.path.join(input_dir, "topic_summary.csv"))

    # For dropdown + labeling
    summary["label"] = summary["topic"].astype(str) + " — " + summary["topic_name"]

    return clusters, summary


# ------------------------------------
# Build Dashboard
# ------------------------------------
def create_dashboard(input_dir="topics_output"):
    clusters, summary = load_data(input_dir)

    app = dash.Dash(__name__)
    app.title = "Topic Model Dashboard"

    app.layout = html.Div([
        html.H1("Interactive Topic Modeling Dashboard"),

        html.Div([
            html.Label("Select Topic:"),
            dcc.Dropdown(
                id="topic-dropdown",
                options=[{"label": lbl, "value": t} for t, lbl in zip(summary["topic"], summary["label"])],
                placeholder="Select a topic",
                clearable=True
            ),
        ], style={"width": "30%", "display": "inline-block"}),

        html.Br(),

        # UMAP Graph
        html.Div([
            html.H2("UMAP Visualization"),
            dcc.Graph(id="umap-plot")
        ]),

        html.Br(),

        # Cluster Size Chart
        html.Div([
            html.H2("Cluster Size Distribution"),
            dcc.Graph(id="cluster-size-chart")
        ]),

        html.Br(),

        # Topic Summary Section
        html.Div([
            html.H2("Topic Summary"),
            html.Div(id="topic-summary-box")
        ]),

        html.Br(),

        # Reviews Table
        html.H2("Reviews under this Topic"),
        dash_table.DataTable(
            id="review-table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal"},
        )
    ], style={"margin": "40px"})

    # -----------------------------------------
    # Callbacks
    # -----------------------------------------

    # Update UMAP plot
    @app.callback(
        Output("umap-plot", "figure"),
        Input("topic-dropdown", "value"))
    def update_umap(selected_topic):
        df = clusters
        if selected_topic is not None:
            df = df[df["topic"] == selected_topic]

        fig = px.scatter(
            df,
            x="umap_x", y="umap_y",
            color="topic",
            hover_data={"review": True, "topic": True},
            title="UMAP Topic Clusters"
        )
        fig.update_layout(height=600)
        return fig

    # Cluster size chart
    @app.callback(
        Output("cluster-size-chart", "figure"),
        Input("topic-dropdown", "value"))
    def update_cluster_plot(_):
        df = clusters.groupby("topic").size().reset_index(name="count")
        fig = px.bar(df, x="topic", y="count", title="Cluster Sizes")
        return fig

    # Topic summary box
    @app.callback(
        Output("topic-summary-box", "children"),
        Input("topic-dropdown", "value"))
    def update_topic_summary(topic_id):
        if topic_id is None:
            return "Select a topic to see details."

        row = summary[summary["topic"] == topic_id].iloc[0]
        return html.Div([
            html.H3(f"{row['topic_name']} (Topic {topic_id})"),
            html.P(f"Top Words: {row['top_words']}"),
        ])

    # Review table
    @app.callback(
        Output("review-table", "data"),
        Output("review-table", "columns"),
        Input("topic-dropdown", "value"))
    def update_review_table(topic_id):
        if topic_id is None:
            df = clusters.sample(min(20, len(clusters)))  # show random small preview
        else:
            df = clusters[clusters["topic"] == topic_id]

        cols = [{"name": c.capitalize(), "id": c} for c in ["review", "cleaned_text", "topic"]]
        return df[["review", "cleaned_text", "topic"]].to_dict("records"), cols

    app.run_server(debug=True)


# ------------------------------------
# MAIN
# ------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="topics_output")
    args = parser.parse_args()

    create_dashboard(args.input_dir)

#python dashboard.py --input_dir topics_output
#http://127.0.0.1:8050
