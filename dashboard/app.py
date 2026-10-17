"""Streamlit investigation dashboard.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudnet.config import ARTIFACT_DIR  # noqa: E402
from fraudnet.models.pipeline import FraudPipeline  # noqa: E402
from fraudnet.risk.scoring import RiskScorer  # noqa: E402

st.set_page_config(page_title="FraudNet · Investigation", page_icon="🛡️", layout="wide")

TIER_COLORS = {"CRITICAL": "#b42318", "HIGH": "#e8590c", "MEDIUM": "#d4a10a", "LOW": "#2f9e44"}
NODE_COLORS = {"card": "#4263eb", "device": "#ae3ec9", "email_payer": "#0c8599",
               "email_recv": "#15aabf", "address": "#e67700", "ip_proxy": "#868e96"}


@st.cache_resource
def load_pipeline() -> FraudPipeline:
    return FraudPipeline.load(ARTIFACT_DIR / "fraud_pipeline.joblib")


@st.cache_data
def load_artifacts():
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())
    valid = pd.read_parquet(ARTIFACT_DIR / "valid_sample.parquet")
    return report, valid


@st.cache_data(show_spinner="Scoring validation window…")
def score_valid(n: int) -> pd.DataFrame:
    _, valid = load_artifacts()
    sample = valid.tail(n).reset_index(drop=True)
    res = pd.DataFrame([r.to_dict() for r in RiskScorer(load_pipeline()).score(sample, explain=False)])
    return pd.concat([sample[["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
                              "card4", "P_emaildomain", "DeviceInfo", "isFraud"]],
                      res.drop(columns=["transaction_id", "top_features"])], axis=1)


def network_figure(sub: dict, height: int = 560) -> go.Figure:
    G = nx.Graph()
    for n in sub["nodes"]:
        G.add_node(n["id"], **n)
    G.add_edges_from((e["source"], e["target"]) for e in sub["edges"])
    pos = nx.spring_layout(G, seed=7, k=1.2 / max(len(G) ** 0.5, 1))
    ex, ey = [], []
    for u, v in G.edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.7, color="#adb5bd"),
                               hoverinfo="skip", showlegend=False))
    for kind, color in NODE_COLORS.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if not nodes:
            continue
        if kind == "card":
            colors = ["#e03131" if G.nodes[n].get("fraud") else color for n in nodes]
            sizes = [16 if n == sub["center"] else 10 for n in nodes]
        else:
            colors, sizes = color, [14 if n == sub["center"] else 12 for n in nodes]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
            name=kind, marker=dict(size=sizes, color=colors, line=dict(width=1, color="white"),
                                   symbol="circle" if kind == "card" else "diamond"),
            text=[f"{n}<br>fraud={G.nodes[n].get('fraud', '-')}"
                  f"<br>proximity={G.nodes[n].get('proximity', 0):.2f}" for n in nodes],
            hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.02))
    return fig


# --------------------------------------------------------------------- app
if not (ARTIFACT_DIR / "fraud_pipeline.joblib").exists():
    st.error("No trained model found. Run `python scripts/run_pipeline.py --synthetic` first.")
    st.stop()

pipe = load_pipeline()
report, valid = load_artifacts()

st.title("🛡️ FraudNet — Fraud Investigation Console")
st.caption("XGBoost transaction model + heterogeneous entity graph for fraud-ring detection")

tab_overview, tab_alerts, tab_rings, tab_explore, tab_score = st.tabs(
    ["Overview", "Alert queue", "Fraud rings", "Graph explorer", "Score a transaction"])

# ---------------------------------------------------------------- overview
with tab_overview:
    m = pipe.metrics
    c = st.columns(5)
    c[0].metric("ROC-AUC (out-of-time)", f"{m['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c[2].metric("Recall @ 90% precision", f"{m['recall_at_90_precision']:.1%}")
    c[3].metric("Graph nodes", f"{pipe.graph.G.number_of_nodes():,}")
    c[4].metric("Suspicious rings", len(pipe.graph.fraud_clusters()))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Ablation: does the graph help?")
        abl = pd.DataFrame(report).T[["roc_auc", "pr_auc", "precision", "recall", "f1",
                                      "recall_at_90_precision"]]
        st.dataframe(abl.style.format("{:.4f}").highlight_max(axis=0, color="#d3f9d8"),
                     width="stretch")
    with right:
        st.subheader("Top features (gain)")
        imp = pipe.feature_importance(20)
        fig = px.bar(imp[::-1], x="gain", y="feature", color="group", orientation="h", height=480,
                     color_discrete_map={"graph": "#ae3ec9", "tabular": "#4263eb"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ alerts
with tab_alerts:
    n = st.slider("Transactions to score (latest in validation window)", 500, min(10_000, len(valid)),
                  min(3000, len(valid)), step=500)
    scored = score_valid(n)
    c = st.columns(4)
    for i, tier in enumerate(["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
        sub = scored[scored.risk_tier == tier]
        hit = f" · {sub.isFraud.mean():.0%} confirmed fraud" if len(sub) else ""
        c[i].metric(tier, f"{len(sub):,}", hit, delta_color="off")

    tiers = st.multiselect("Tiers", list(TIER_COLORS), ["CRITICAL", "HIGH"])
    q = scored[scored.risk_tier.isin(tiers)].sort_values("risk_score", ascending=False)
    q = q.assign(reasons=q.reasons.map(lambda r: "; ".join(r)))
    st.dataframe(q[["TransactionID", "risk_score", "risk_tier", "model_probability", "graph_risk",
                    "TransactionAmt", "ProductCD", "P_emaildomain", "DeviceInfo", "reasons",
                    "isFraud"]], width="stretch", height=420,
                 column_config={"risk_score": st.column_config.ProgressColumn(min_value=0, max_value=1),
                                "isFraud": st.column_config.CheckboxColumn("label")})
    fig = px.histogram(scored, x="risk_score", color=scored.isFraud.map({0: "legit", 1: "fraud"}),
                       nbins=50, log_y=True, barmode="overlay", height=280,
                       color_discrete_map={"legit": "#4263eb", "fraud": "#e03131"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------- rings
with tab_rings:
    c1, c2 = st.columns(2)
    min_cards = c1.slider("Min cards in cluster", 2, 30, 3)
    min_rate = c2.slider("Min fraud rate", 0.0, 1.0, 0.3, 0.05)
    clusters = pipe.graph.fraud_clusters(min_cards, min_rate, top_k=100)
    if not clusters:
        st.info("No clusters match these filters.")
    else:
        cdf = pd.DataFrame([{**c.to_dict(), "shared_entities": ", ".join(c.shared_entities[:6])}
                            for c in clusters]).drop(columns="cards")
        st.dataframe(cdf, width="stretch", height=260,
                     column_config={"fraud_rate": st.column_config.ProgressColumn(min_value=0, max_value=1),
                                    "total_amount": st.column_config.NumberColumn(format="$%.0f")})
        pick = st.selectbox("Inspect cluster", [c.cluster_id for c in clusters],
                            format_func=lambda i: f"Cluster {i}")
        chosen = next(c for c in clusters if c.cluster_id == pick)
        st.plotly_chart(network_figure(pipe.graph.subgraph(chosen.cards[0], radius=2)),
                        width="stretch")
        st.caption("Red cards = confirmed fraud · diamonds = shared entities "
                   "(device, e-mail, address, proxy)")

# ---------------------------------------------------------------- explorer
with tab_explore:
    examples = [c.shared_entities[0] for c in pipe.graph.fraud_clusters(top_k=5) if c.shared_entities]
    node = st.text_input("Node id (e.g. `device:…`, `card:…`, `address:…`)",
                         value=examples[0] if examples else "")
    radius = st.radio("Radius", [1, 2], index=1, horizontal=True)
    if node:
        try:
            sub = pipe.graph.subgraph(node, radius=radius)
            st.plotly_chart(network_figure(sub), width="stretch")
            nodes = pd.DataFrame(sub["nodes"])
            st.dataframe(nodes, width="stretch", height=240)
        except KeyError:
            st.warning(f"`{node}` is not in the graph.")

# ------------------------------------------------------------------- score
with tab_score:
    st.write("Score a single transaction with full explanation.")
    base = valid.sample(1, random_state=st.session_state.get("seed", 0)).iloc[0]
    if st.button("🎲 Random transaction from validation set"):
        st.session_state["seed"] = st.session_state.get("seed", 0) + 1
        st.rerun()
    c = st.columns(4)
    amt = c[0].number_input("Amount ($)", 0.01, 50_000.0, float(base.TransactionAmt))
    prod = c[1].selectbox("ProductCD", ["W", "C", "R", "H", "S"],
                          index=["W", "C", "R", "H", "S"].index(base.ProductCD))
    email = c[2].text_input("Payer e-mail domain", str(base.P_emaildomain))
    device = c[3].text_input("DeviceInfo", "" if pd.isna(base.DeviceInfo) else str(base.DeviceInfo))
    tx = base.to_frame().T.copy()
    tx["TransactionAmt"], tx["ProductCD"], tx["P_emaildomain"] = amt, prod, email
    tx["DeviceInfo"] = device or None
    tx = tx.infer_objects()
    r = RiskScorer(pipe).score(tx)[0]

    c = st.columns(4)
    c[0].markdown(f"### <span style='color:{TIER_COLORS[r.risk_tier]}'>{r.risk_tier}</span>",
                  unsafe_allow_html=True)
    c[1].metric("Risk score", f"{r.risk_score:.3f}")
    c[2].metric("Model probability", f"{r.model_probability:.3f}")
    c[3].metric("Graph risk", f"{r.graph_risk:.3f}")
    if r.reasons:
        st.markdown("**Why:** " + " · ".join(r.reasons))
    contrib = pd.DataFrame(r.top_features)
    fig = px.bar(contrib[::-1], x="contribution", y="feature", orientation="h", height=260,
                 color=contrib[::-1].contribution > 0,
                 color_discrete_map={True: "#e03131", False: "#2f9e44"})
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                      xaxis_title="contribution to log-odds (TreeSHAP)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Ground truth label: {'fraud' if base.isFraud else 'legit'}")
