"""Dash dashboard for the existing offline V2X fusion simulation."""

from functools import lru_cache
import math

from dash import Dash, Input, Output, dcc, html, dash_table
import plotly.graph_objects as go

from src.simulation import run_simulation


SCENARIOS = {
    "标准场景": (100, 5),
    "遮挡补盲": (300, 10),
    "通信退化": (500, 30),
}

app = Dash(__name__, title="V2X 协同感知融合 Demo")
server = app.server


@lru_cache(maxsize=64)
def simulation(delay_ms, loss_percent, occlusion, adaptive, seed):
    return run_simulation(
        delay_s=delay_ms / 1000,
        loss_rate=loss_percent / 100,
        occlusion=occlusion,
        adaptive=adaptive,
        seed=seed,
    )


def option(label, component):
    return html.Div([html.Label(label), component], style={"marginBottom": "18px"})


def card(label, value, subtitle=""):
    return html.Div(
        [html.Div(label, style={"color": "#475569"}),
         html.H2(value, style={"margin": "8px 0"}),
         html.Small(subtitle)],
        style={"background": "white", "padding": "18px", "borderRadius": "12px"},
    )


app.layout = html.Div(
    [
        html.H1("面向遮挡交叉口的车路协同感知融合"),
        html.P("离线可复现仿真｜目标级融合｜时间补偿｜匈牙利关联｜通信退化保护"),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("实验设置"),
                        option("预设场景", dcc.Dropdown(
                            list(SCENARIOS), "遮挡补盲", id="scenario", clearable=False)),
                        option("路侧通信时延（ms）", dcc.Slider(
                            0, 500, 50, value=300, id="delay", marks={i: str(i) for i in range(0, 501, 100)})),
                        option("路侧丢包率（%）", dcc.Slider(
                            0, 30, 5, value=10, id="loss", marks={i: str(i) for i in range(0, 31, 10)})),
                        option("车端遮挡", dcc.Checklist(
                            [{"label": "启用", "value": "on"}], ["on"], id="occlusion")),
                        option("自适应权重", dcc.Checklist(
                            [{"label": "启用", "value": "on"}], ["on"], id="adaptive")),
                        option("随机种子", dcc.Input(
                            id="seed", type="number", min=0, max=9999, step=1, value=42)),
                        html.P("固定种子便于复现实验；下方消融使用相同观测与通信条件。"),
                    ],
                    style={"background": "white", "padding": "20px", "borderRadius": "12px"},
                ),
                html.Div(
                    [
                        html.Div(id="metrics", style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "12px"}),
                        html.H3("逐帧回放"),
                        dcc.Slider(0, 120, 1, value=60, id="frame", updatemode="mouseup"),
                        html.Div(
                            [dcc.Graph(id="trajectory", style={"flex": "1.7", "minWidth": "360px"}),
                             html.Div(id="frame-info", style={"flex": "1", "minWidth": "250px", "background": "white", "padding": "18px", "borderRadius": "12px"})],
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "14px"},
                        ),
                        html.H3("融合权重变化"),
                        dcc.Graph(id="weights"),
                        html.H3("固定权重与自适应权重消融"),
                        dash_table.DataTable(
                            id="comparison",
                            columns=[{"name": name, "id": name} for name in
                                     ["融合策略", "融合RMSE / m", "相对单车改善 / %", "双端帧平均路侧权重"]],
                            style_cell={"padding": "12px", "textAlign": "left"},
                            style_header={"backgroundColor": "#e2e8f0", "fontWeight": "bold"},
                        ),
                    ],
                    style={"minWidth": "0"},
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "minmax(230px, 270px) minmax(0, 1fr)", "gap": "20px"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "color": "#0f172a", "background": "#f8fafc", "padding": "24px", "minHeight": "100vh"},
)


@app.callback(Output("delay", "value"), Output("loss", "value"), Input("scenario", "value"))
def select_scenario(name):
    return SCENARIOS[name]


@app.callback(
    Output("metrics", "children"), Output("frame", "max"),
    Output("trajectory", "figure"), Output("frame-info", "children"),
    Output("weights", "figure"), Output("comparison", "data"),
    Input("delay", "value"), Input("loss", "value"), Input("occlusion", "value"),
    Input("adaptive", "value"), Input("seed", "value"), Input("frame", "value"),
)
def update_dashboard(delay, loss, occlusion, adaptive, seed, frame):
    delay = int(delay or 0)
    loss = int(loss or 0)
    seed = max(0, min(9999, int(seed or 0)))
    occlusion = "on" in (occlusion or [])
    adaptive = "on" in (adaptive or [])
    current = simulation(delay, loss, occlusion, adaptive, seed)
    fixed = simulation(delay, loss, occlusion, False, seed)
    adapted = simulation(delay, loss, occlusion, True, seed)
    frames, events, metrics = current.frames, current.events, current.metrics
    index = max(0, min(int(frame or 0), len(frames) - 1))
    row, event = frames.iloc[index], events.iloc[index]
    history = frames.iloc[:index + 1]

    cards = [
        card("单车位置 RMSE", f"{metrics['vehicle_rmse']:.4f} m"),
        card("融合位置 RMSE", f"{metrics['fused_rmse']:.4f} m", f"改善 {metrics['improvement_pct']:.2f}%"),
        card("车端漏检率", f"{metrics['vehicle_miss_rate']:.2%}"),
        card("路侧接收率", f"{metrics['roadside_receive_rate']:.2%}"),
    ]

    trajectory = go.Figure()
    for x0, x1, y0, y1, color in [(-45, 60, -7, 7, "#94a3b8"),
                                   (-7, 7, -25, 25, "#94a3b8"),
                                   (8, 45, 8, 25, "#475569")]:
        trajectory.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                             fillcolor=color, opacity=0.25, line_width=0, layer="below")
    trajectory.add_annotation(x=26, y=16, text="建筑遮挡区", showarrow=False)
    series = [
        ("Ground Truth", "truth", "lines", "#111827"),
        ("车端检测", "vehicle_measurement", "markers", "#2563eb"),
        ("路侧检测", "roadside_measurement", "markers", "#f59e0b"),
        ("单车估计", "vehicle_estimate", "lines", "#2563eb"),
        ("融合估计", "fused_estimate", "lines+markers", "#16a34a"),
    ]
    for name, prefix, mode, color in series:
        trajectory.add_trace(go.Scatter(
            x=history[f"{prefix}_x"], y=history[f"{prefix}_y"], name=name,
            mode=mode, line={"color": color, "width": 3, "dash": "dot" if prefix == "vehicle_estimate" else "solid"},
            marker={"color": color, "size": 6},
        ))
    trajectory.update_layout(height=480, margin=dict(l=20, r=20, t=35, b=30),
                             xaxis_title="X 位置 / m", yaxis_title="Y 位置 / m",
                             legend=dict(orientation="h", y=1.12))
    trajectory.update_yaxes(scaleanchor="x", scaleratio=1)

    status = "｜".join([
        "位于遮挡区域" if bool(row["occluded"]) else "非遮挡区域",
        "车端有效" if bool(row["vehicle_available"]) else "车端漏检",
        "路侧有效" if bool(row["roadside_available"]) else "路侧丢包",
    ])
    distance = row["association_distance"]
    info = [html.H3("当前帧信息"), html.P(f"t = {row['time']:.1f} s｜目标 ID 101｜{status}"),
            html.Strong("当前处理流程"), html.P(str(event["event"])),
            html.P(f"路侧融合权重：{row['roadside_weight']:.4f}"),
            html.Progress(value=float(row["roadside_weight"]), max=1, style={"width": "100%"}),
            html.P(f"真实位置：({row['truth_x']:.2f}, {row['truth_y']:.2f}) m"),
            html.P(f"单车估计：({row['vehicle_estimate_x']:.2f}, {row['vehicle_estimate_y']:.2f}) m"),
            html.P(f"融合估计：({row['fused_estimate_x']:.2f}, {row['fused_estimate_y']:.2f}) m"),
            html.P(f"关联距离：{distance:.4f} m" if not math.isnan(distance) else "关联距离：无双端关联"),
            html.P(f"设置的通信时延：{delay} ms")]

    weights = go.Figure()
    for name, column, color in [("车端权重", "vehicle_weight", "#2563eb"),
                                 ("路侧权重", "roadside_weight", "#f59e0b")]:
        weights.add_trace(go.Scatter(x=frames["time"], y=frames[column], mode="lines", name=name,
                                     line={"color": color, "width": 2}))
    if occlusion:
        weights.add_vrect(x0=4, x1=7, fillcolor="#dc2626", opacity=0.08,
                          line_width=0, annotation_text="车端遮挡区间")
    weights.update_layout(height=320, xaxis_title="时间 / s", yaxis_title="融合权重",
                          yaxis_range=[0, 1], margin=dict(l=30, r=20, t=30, b=30))

    comparison = [
        {"融合策略": name, "融合RMSE / m": f"{result.metrics['fused_rmse']:.4f}",
         "相对单车改善 / %": f"{result.metrics['improvement_pct']:.2f}",
         "双端帧平均路侧权重": f"{result.metrics['mean_paired_roadside_weight']:.4f}"}
        for name, result in [("固定权重", fixed), ("自适应权重", adapted)]
    ]
    return cards, len(frames) - 1, trajectory, info, weights, comparison


if __name__ == "__main__":
    app.run(debug=False)
