"""DeGroot learning model - interactive Streamlit app.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="DeGroot Learning Model", page_icon="🕸️", layout="wide")

PALETTE = px.colors.qualitative.Dark24


# ──────────────────────────────────────────────────────────────────────────────
# Maths helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_value(text: str) -> float:
    """Parse '0.25', '1/4' or ' 3 ' into a float."""
    return float(Fraction(text.strip()))


def normalize_rows(W) -> np.ndarray:
    """Clip negatives / NaNs, then make every row sum to 1.

    A row that sums to 0 becomes a 'stubborn' row (agent only trusts itself).
    """
    W = np.clip(np.nan_to_num(np.asarray(W, dtype=float)), 0, None)
    zero_rows = W.sum(axis=1) == 0
    W[zero_rows] = np.eye(len(W))[zero_rows]
    return W / W.sum(axis=1, keepdims=True)


def simulate(W: np.ndarray, x0: np.ndarray, T: int) -> np.ndarray:
    X = np.empty((T + 1, len(x0)))
    X[0] = x0
    for t in range(T):
        X[t + 1] = W @ X[t]
    return X


def analyse(W: np.ndarray, x0: np.ndarray) -> dict:
    """Spectral properties + long-run behaviour of x(t+1) = W x(t)."""
    n = len(W)
    eig = np.linalg.eigvals(W)
    mods = np.sort(np.abs(eig))[::-1]
    A = np.linalg.matrix_power(W, 5000)  # proxy for lim W^t
    limit_exists = bool(np.allclose(A, A @ W, atol=1e-5))
    consensus = limit_exists and bool(np.allclose(A, A[0], atol=1e-4))
    slem = float(mods[1]) if n > 1 else 0.0
    steps99 = None
    if consensus and 0 < slem < 1:
        steps99 = int(np.ceil(np.log(0.01) / np.log(slem)))
    return dict(
        eig=eig,
        rho=float(mods[0]),
        slem=slem,
        limit_exists=limit_exists,
        consensus=consensus,
        limit=(A @ x0) if limit_exists else None,
        influence=A[0] if consensus else None,
        steps99=steps99,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Scenario presets
# ──────────────────────────────────────────────────────────────────────────────
def preset_random(n, rng):
    return rng.random((n, n)), rng.random(n)


def preset_consensus(n, rng):
    return np.full((n, n), 1 / n), np.linspace(0, 1, n)


def preset_polarised(n, rng):
    h = max(n // 2, 1)
    W = np.full((n, n), 0.02)
    W[:h, :h] = 1
    W[h:, h:] = 1
    x0 = np.r_[np.full(h, 0.1), np.full(n - h, 0.9)] + rng.normal(0, 0.02, n)
    return W, x0


def preset_stubborn(n, rng):
    W = rng.random((n, n)) + 0.1
    W[0] = 0
    W[0, 0] = 1  # agent 1 never changes its mind
    x0 = rng.random(n)
    x0[0] = 1.0
    return W, x0


def preset_ring(n, rng):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] += 0.5
        W[i, (i + 1) % n] += 0.5
    return W, rng.random(n)


def preset_influencer(n, rng):
    W = rng.random((n, n)) * 0.2
    W[:, 0] += 1.0  # everybody listens mostly to agent 1
    W[0] = rng.random(n)  # agent 1 listens to everybody equally-ish
    return W, rng.random(n)


PRESETS = {
    "🎲 Random": preset_random,
    "🤝 Consensus": preset_consensus,
    "⚔️ Polarised": preset_polarised,
    "🧱 Stubborn agent": preset_stubborn,
    "🔄 Ring": preset_ring,
    "👑 Influencer": preset_influencer,
}


# ──────────────────────────────────────────────────────────────────────────────
# Session state + callbacks
# ──────────────────────────────────────────────────────────────────────────────
st.session_state.setdefault("n", 5)
st.session_state.setdefault("T", 30)
st.session_state.setdefault("ver", 0)
st.session_state.setdefault("result", None)
st.session_state.setdefault("paste_error", "")


def set_state(W, x0):
    st.session_state.W = normalize_rows(W)
    st.session_state.x0 = np.asarray(x0, dtype=float)
    st.session_state.ver += 1  # forces the data editors to reload
    st.session_state.result = None


def apply_preset(label: str):
    W, x0 = PRESETS[label](st.session_state.n, np.random.default_rng())
    set_state(W, x0)


def load_pasted():
    try:
        rows = [
            [parse_value(v) for v in line.replace(",", " ").split()]
            for line in st.session_state.paste_W.strip().splitlines()
            if line.strip()
        ]
        x0 = [parse_value(v) for v in st.session_state.paste_x.replace(",", " ").split()]
        W = np.array(rows, dtype=float)
        n = len(rows)
        if W.shape != (n, n):
            raise ValueError(f"Matrix must be square, got {W.shape}.")
        if len(x0) != n:
            raise ValueError(f"Need {n} opinions, got {len(x0)}.")
        if not 2 <= n <= 20:
            raise ValueError("Number of agents must be between 2 and 20.")
        st.session_state.n = n
        set_state(W, x0)
        st.session_state.paste_error = ""
    except Exception as exc:  # noqa: BLE001
        st.session_state.paste_error = f"Could not parse input: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Plot builders
# ──────────────────────────────────────────────────────────────────────────────
def fig_heatmap(W, short):
    n = len(W)
    s = min(11, 3.5 + 0.55 * n)
    fig, ax = plt.subplots(figsize=(s, s * 0.85))
    sns.heatmap(
        W, annot=n <= 12, fmt=".2f", cmap="YlGnBu", vmin=0, ax=ax, square=True,
        xticklabels=short, yticklabels=short, cbar_kws={"label": "trust weight"},
        annot_kws={"size": 10},
    )
    ax.set_xlabel("Trust target (who is listened to)")
    ax.set_ylabel("Trust source (who listens)")
    ax.set_title("Trust matrix W", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_network(W, threshold, edge_labels):
    n = len(W)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] >= threshold:
                G.add_edge(i, j, weight=W[i, j])
    pos = nx.circular_layout(G)
    self_trust = np.diag(W)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=700 + 2200 * self_trust, node_color=self_trust,
        cmap="Blues", vmin=0, vmax=1, edgecolors="black",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=9, font_weight="bold",
        labels={i: f"A{i+1}\n{self_trust[i]:.2f}" for i in range(n)},
    )
    widths = [0.5 + 7 * G[u][v]["weight"] for u, v in G.edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, arrowstyle="-|>", arrowsize=16, edge_color="gray",
        connectionstyle="arc3,rad=0.12", node_size=700 + 2200 * self_trust,
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, font_size=8, font_color="darkred", label_pos=0.3,
            edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)},
            connectionstyle="arc3,rad=0.12",
        )
    ax.set_title("Trust network  (arrow i → j: i listens to j;  node label = self-trust)",
                 fontweight="bold", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_spectrum(eig, rho):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "--", color="gray", label="Unit circle")
    ax.scatter(eig.real, eig.imag, color="crimson", s=70, zorder=3, label="Eigenvalues")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_aspect("equal")
    ax.grid(True, ls=":")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title(f"Eigenvalues of W  (spectral radius = {rho:.4f})", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_convergence(X, labels, limit_value):
    fig = go.Figure()
    for i, lab in enumerate(labels):
        fig.add_scatter(x=np.arange(len(X)), y=X[:, i], mode="lines+markers", name=lab,
                        line=dict(color=PALETTE[i % len(PALETTE)]))
    if limit_value is not None:
        fig.add_hline(y=limit_value, line_dash="dash", line_color="black",
                      annotation_text=f"consensus ≈ {limit_value:.3f}")
    fig.update_layout(xaxis_title="Time step t", yaxis_title="Opinion",
                      height=480, margin=dict(t=30))
    return fig


def anim_controls(T1, duration):
    """Play / Pause / Restart buttons + time slider shared by all animations."""
    still = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
             "transition": {"duration": 0}}
    menus = [dict(
        type="buttons", direction="left", x=0.0, y=1.14, xanchor="left", yanchor="top",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": duration, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}]),
            dict(label="⏮ Restart", method="animate", args=[["0"], still]),
        ])]
    sliders = [dict(active=0, currentvalue={"prefix": "t = "}, pad={"t": 40},
                    steps=[dict(method="animate", label=str(t), args=[[str(t)], still])
                           for t in range(T1)])]
    return menus, sliders


def fig_animation(X, labels, mode, duration):
    T1, n = X.shape
    lo, hi = float(X.min()), float(X.max())
    pad = 0.08 * ((hi - lo) or 1.0)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    if mode == "Lines":
        def traces(t):
            return [go.Scatter(x=list(range(t + 1)), y=X[: t + 1, i], mode="lines+markers",
                               name=labels[i], line=dict(color=colors[i])) for i in range(n)]
        xaxis = dict(range=[0, T1 - 1], title="Time step t")
    else:
        def traces(t):
            return [go.Bar(x=labels, y=X[t], marker_color=colors, showlegend=False)]
        xaxis = dict(title="Agent")

    fig = go.Figure(data=traces(0),
                    frames=[go.Frame(data=traces(t), name=str(t)) for t in range(T1)])
    menus, sliders = anim_controls(T1, duration)
    fig.update_layout(
        height=560, xaxis=xaxis,
        yaxis=dict(range=[lo - pad, hi + pad], title="Opinion"),
        margin=dict(t=90), updatemenus=menus, sliders=sliders,
    )
    return fig


def fig_network_animation(W, X, short, threshold, duration):
    """Trust network whose node colours follow each agent's opinion over time."""
    T1, n = X.shape
    ang = 2 * np.pi * np.arange(n) / n
    node_x, node_y = np.cos(ang), np.sin(ang)
    lo, hi = float(X.min()), float(X.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5

    edge_traces = []
    for w_lo, w_hi, width, alpha in [(0.0, 0.1, 1.0, 0.25), (0.1, 0.25, 2.5, 0.4),
                                     (0.25, 1.01, 5.0, 0.6)]:
        xs, ys = [], []
        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] >= threshold and w_lo <= W[i, j] < w_hi:
                    xs += [node_x[i], node_x[j], None]
                    ys += [node_y[i], node_y[j], None]
        edge_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(width=width, color=f"rgba(110,110,110,{alpha})")))

    def node_trace(t):
        return go.Scatter(
            x=node_x, y=node_y, mode="markers+text", showlegend=False,
            text=[f"{short[i]}<br>{X[t, i]:.2f}" for i in range(n)],
            textfont=dict(size=10, color="black"),
            marker=dict(size=48, color=X[t], cmin=lo, cmax=hi,
                        colorscale=[[0, "#fff7bc"], [0.5, "#fdae61"], [1, "#d7301f"]],
                        line=dict(width=1, color="black"),
                        colorbar=dict(title="Opinion")),
            hovertemplate="%{text}<extra></extra>")

    k = len(edge_traces)
    fig = go.Figure(
        data=edge_traces + [node_trace(0)],
        frames=[go.Frame(data=[node_trace(t)], traces=[k], name=str(t)) for t in range(T1)],
    )
    menus, sliders = anim_controls(T1, duration)
    fig.update_layout(
        height=680, margin=dict(t=90), updatemenus=menus, sliders=sliders,
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    st.slider("Number of agents", 2, 20, key="n")
    st.slider("Time steps", 1, 200, key="T")

    st.subheader("Load a scenario")
    c1, c2 = st.columns(2)
    for idx, name in enumerate(PRESETS):
        (c1 if idx % 2 == 0 else c2).button(name, on_click=apply_preset, args=(name,))

    with st.expander("📋 Paste your own data"):
        st.caption("One matrix row per line, values separated by spaces or commas. "
                   "Fractions like 1/3 work.")
        st.text_area("Trust matrix W", key="paste_W", height=130,
                     placeholder="1/2 1/2 0\n0 1/2 1/2\n1/3 1/3 1/3")
        st.text_area("Initial opinions x(0)", key="paste_x", height=70,
                     placeholder="0.2 0.5 0.9")
        st.button("📥 Load pasted data", on_click=load_pasted)
        if st.session_state.paste_error:
            st.error(st.session_state.paste_error)

n = st.session_state.n
if "W" not in st.session_state or st.session_state.W.shape[0] != n:
    set_state(*preset_random(n, np.random.default_rng()))

labels = [f"Agent {i+1}" for i in range(n)]
short = [f"A{i+1}" for i in range(n)]

# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────
st.title("🕸️ DeGroot Learning Model")
st.caption("Agents repeatedly replace their opinion by a trust-weighted average of everyone's "
           "opinions:  x(t+1) = W · x(t).")

st.subheader("1 · Inputs")
ver = st.session_state.ver
left, right = st.columns([3, 1])
with left:
    st.markdown("**Trust matrix W**  · row *i* = how much agent *i* trusts each agent (click a cell to edit)")
    W_edit = st.data_editor(
        pd.DataFrame(st.session_state.W, index=labels, columns=labels),
        key=f"W_{ver}",
        column_config={c: st.column_config.NumberColumn(c, min_value=0.0, step=0.01, format="%.3f")
                       for c in labels},
    )
with right:
    st.markdown("**Initial opinions x(0)**")
    x_edit = st.data_editor(
        pd.DataFrame({"x(0)": st.session_state.x0}, index=labels),
        key=f"x_{ver}",
        column_config={"x(0)": st.column_config.NumberColumn("x(0)", step=0.01, format="%.3f")},
    )

W_in = normalize_rows(W_edit.to_numpy(dtype=float))
x_in = np.nan_to_num(x_edit.iloc[:, 0].to_numpy(dtype=float))

raw_sums = np.nan_to_num(W_edit.to_numpy(dtype=float)).sum(axis=1)
if not np.allclose(raw_sums, 1.0, atol=1e-3):
    st.info("Some rows of W don't sum to 1 - they are normalised automatically when you run.")

b1, b2, _ = st.columns([1, 1, 4])
run = b1.button("▶ Run simulation", type="primary")
if b2.button("⚖️ Normalise table"):
    st.session_state.W = W_in
    st.session_state.x0 = x_in
    st.session_state.ver += 1
    st.rerun()

if run or st.session_state.result is None:
    T = st.session_state.T
    st.session_state.result = dict(
        W=W_in, x0=x_in, T=T, X=simulate(W_in, x_in, T), labels=labels, short=short,
        **analyse(W_in, x_in),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
R = st.session_state.result
W, X, eig, T = R["W"], R["X"], R["eig"], R["T"]
labels, short = R["labels"], R["short"]
n = len(W)

st.subheader("2 · Results")
tabs = st.tabs(["📊 Overview", "🔥 Trust matrix", "🕸️ Network graph", "📈 Convergence graph",
                "🎬 Convergence animation", "🌐 Network animation", "🌀 Spectrum", "🗂️ Data"])

# ── Overview ──
with tabs[0]:
    if R["consensus"]:
        status = "Consensus ✅"
    elif R["limit_exists"]:
        status = "Clusters (no consensus) ⚠️"
    else:
        status = "Oscillates ❌"
    spread_final = X[-1].max() - X[-1].min()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Long-run behaviour", status)
    m2.metric("Spectral radius ρ(W)", f"{R['rho']:.4f}")
    m3.metric("Second eigenvalue |λ₂|", f"{R['slem']:.4f}",
              help="The closer to 1, the slower opinions converge.")
    m4.metric(f"Disagreement at t={T}", f"{spread_final:.4f}",
              delta=f"{spread_final - (X[0].max() - X[0].min()):+.4f} vs t=0", delta_color="inverse")

    if R["consensus"]:
        msg = f"Everyone converges to **{R['limit'][0]:.4f}**."
        if R["steps99"]:
            msg += f" Disagreement shrinks by 99% in roughly **{R['steps99']} steps**."
        st.success(msg)
        infl = px.bar(x=labels, y=R["influence"], labels={"x": "", "y": "influence weight"},
                      title="Social influence: weight each agent has on the final consensus")
        infl.update_layout(height=340, showlegend=False)
        st.plotly_chart(infl)
    elif R["limit_exists"]:
        st.warning("Opinions settle, but different groups settle at different values "
                   "(W has several closed groups of agents that ignore each other).")
    else:
        st.error("Opinions do not settle (periodic structure, e.g. a pure cycle with no self-trust).")

    summary = pd.DataFrame({"x(0)": R["x0"], f"x({T})": X[-1]}, index=labels)
    if R["limit"] is not None:
        summary["x(∞)"] = R["limit"]
    st.dataframe(summary.round(4))

# ── Trust matrix heatmap ──
with tabs[1]:
    fig = fig_heatmap(W, short)
    st.pyplot(fig)
    plt.close(fig)

# ── Static network graph ──
with tabs[2]:
    k1, k2 = st.columns(2)
    thr = k1.slider("Hide edges lighter than", 0.0, 0.5, 0.0 if n <= 6 else 0.05, 0.01,
                    key="thr_static")
    show_lab = k2.toggle("Show edge weights", value=n <= 6, key="lab_static")
    fig = fig_network(W, thr, show_lab)
    st.pyplot(fig)
    plt.close(fig)

# ── Static convergence graph ──
with tabs[3]:
    cons = float(R["limit"][0]) if R["consensus"] else None
    st.plotly_chart(fig_convergence(X, labels, cons))

# ── Convergence animation (lines / bars) ──
with tabs[4]:
    a1, a2 = st.columns([1, 2])
    mode = a1.radio("View", ["Lines", "Bars"], horizontal=True, key="anim_mode")
    dur = a2.slider("Frame duration (ms)", 50, 1000, 250, 50, key="dur_conv")
    st.plotly_chart(fig_animation(X, labels, mode, dur))
    st.caption("Use ▶ Play / ⏸ Pause / ⏮ Restart or drag the slider to scrub through time.")

# ── Network animation (node colour = opinion) ──
with tabs[5]:
    n1, n2 = st.columns([2, 2])
    thr_a = n1.slider("Hide edges lighter than", 0.0, 0.5, 0.0 if n <= 6 else 0.05, 0.01,
                      key="thr_anim")
    dur_n = n2.slider("Frame duration (ms)", 50, 1000, 400, 50, key="dur_net")
    st.plotly_chart(fig_network_animation(W, X, short, thr_a, dur_n))
    st.caption("Node colour and label show each agent's opinion at time t; "
               "line thickness shows trust. Watch the colours merge as the group converges.")

# ── Spectrum ──
with tabs[6]:
    fig = fig_spectrum(eig, R["rho"])
    st.pyplot(fig)
    plt.close(fig)
    st.caption("A row-stochastic matrix always has eigenvalue 1. If every other eigenvalue lies "
               "strictly inside the unit circle, opinions reach consensus.")

# ── Data ──
with tabs[7]:
    df_ops = pd.DataFrame(X, columns=labels)
    df_ops.index.name = "t"
    st.dataframe(df_ops.round(5))
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Opinions CSV", df_ops.to_csv().encode(), "opinions.csv", "text/csv")
    d2.download_button("⬇️ Trust matrix CSV",
                       pd.DataFrame(W, index=labels, columns=labels).to_csv().encode(),
                       "trust_matrix.csv", "text/csv")
