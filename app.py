import plotly.graph_objects as go
import streamlit as st

from src.simulation import run_simulation


st.set_page_config(
    page_title="V2X协同感知融合Demo",
    page_icon="🚦",
    layout="wide",
)


SCENARIOS = {
    "标准场景": {
        "delay_ms": 100,
        "loss_percent": 5,
        "description": (
            "轻度通信时延与丢包，"
            "展示常规车路协同融合。"
        ),
    },
    "遮挡补盲": {
        "delay_ms": 300,
        "loss_percent": 10,
        "description": (
            "目标进入车端遮挡区域，"
            "路侧感知提供补盲信息。"
        ),
    },
    "通信退化": {
        "delay_ms": 500,
        "loss_percent": 30,
        "description": (
            "高时延、高丢包条件下，"
            "观察自适应权重降级保护。"
        ),
    },
}


st.title("面向遮挡交叉口的车路协同感知融合")

st.caption(
    "离线可复现仿真｜目标级融合｜"
    "卡尔曼滤波｜匈牙利关联｜通信退化保护"
)


with st.sidebar:
    st.header("实验设置")

    scenario_name = st.selectbox(
        label="预设场景",
        options=list(SCENARIOS.keys()),
        index=1,
    )

    selected_scenario = SCENARIOS[
        scenario_name
    ]

    st.info(
        selected_scenario["description"]
    )

    delay_ms = st.slider(
        label="路侧通信时延（ms）",
        min_value=0,
        max_value=500,
        value=selected_scenario["delay_ms"],
        step=50,
    )

    loss_percent = st.slider(
        label="路侧丢包率（%）",
        min_value=0,
        max_value=30,
        value=selected_scenario[
            "loss_percent"
        ],
        step=5,
    )

    occlusion = st.toggle(
        label="启用车端遮挡",
        value=True,
    )

    adaptive = st.toggle(
        label="启用自适应权重",
        value=True,
    )

    seed = st.number_input(
        label="随机种子",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    st.caption(
        "固定随机种子可保证结果重复。"
    )


result = run_simulation(
    delay_s=delay_ms / 1000.0,
    loss_rate=loss_percent / 100.0,
    occlusion=occlusion,
    adaptive=adaptive,
    seed=int(seed),
)

frames = result.frames
events = result.events
metrics = result.metrics


st.subheader("总体评价指标")

metric_column_1, metric_column_2, \
metric_column_3, metric_column_4 = (
    st.columns(4)
)

metric_column_1.metric(
    label="单车位置RMSE",
    value=(
        f"{metrics['vehicle_rmse']:.4f} m"
    ),
)

metric_column_2.metric(
    label="融合位置RMSE",
    value=(
        f"{metrics['fused_rmse']:.4f} m"
    ),
    delta=(
        f"{metrics['improvement_pct']:.2f}% 改善"
    ),
)

metric_column_3.metric(
    label="车端漏检率",
    value=(
        f"{metrics['vehicle_miss_rate']:.2%}"
    ),
)

metric_column_4.metric(
    label="路侧接收率",
    value=(
        f"{metrics['roadside_receive_rate']:.2%}"
    ),
)


st.divider()

frame_index = st.slider(
    label="拖动滑块回放仿真帧",
    min_value=0,
    max_value=len(frames) - 1,
    value=min(60, len(frames) - 1),
    step=1,
)

current_frame = frames.iloc[frame_index]
current_event = events.iloc[frame_index]


left_column, right_column = st.columns(
    [1.7, 1.0]
)


with left_column:
    st.subheader("交叉口俯视轨迹")

    history = frames.iloc[
        : frame_index + 1
    ]

    trajectory_figure = go.Figure()

    # 横向道路
    trajectory_figure.add_shape(
        type="rect",
        x0=-45,
        x1=60,
        y0=-7,
        y1=7,
        fillcolor="#64748B",
        opacity=0.20,
        line_width=0,
        layer="below",
    )

    # 纵向道路
    trajectory_figure.add_shape(
        type="rect",
        x0=-7,
        x1=7,
        y0=-25,
        y1=25,
        fillcolor="#64748B",
        opacity=0.20,
        line_width=0,
        layer="below",
    )

    # 建筑遮挡区域
    trajectory_figure.add_shape(
        type="rect",
        x0=8,
        x1=45,
        y0=8,
        y1=25,
        fillcolor="#475569",
        opacity=0.35,
        line_width=0,
        layer="below",
    )

    trajectory_figure.add_annotation(
        x=26,
        y=16,
        text="建筑遮挡区",
        showarrow=False,
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=history["truth_x"],
            y=history["truth_y"],
            mode="lines",
            name="Ground Truth",
            line={
                "color": "#111827",
                "width": 3,
            },
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=history[
                "vehicle_measurement_x"
            ],
            y=history[
                "vehicle_measurement_y"
            ],
            mode="markers",
            name="车端检测",
            marker={
                "color": "#2563EB",
                "size": 6,
                "symbol": "circle",
            },
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=history[
                "roadside_measurement_x"
            ],
            y=history[
                "roadside_measurement_y"
            ],
            mode="markers",
            name="路侧检测",
            marker={
                "color": "#F59E0B",
                "size": 7,
                "symbol": "diamond",
            },
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=history[
                "vehicle_estimate_x"
            ],
            y=history[
                "vehicle_estimate_y"
            ],
            mode="lines",
            name="单车估计",
            line={
                "color": "#2563EB",
                "width": 2,
                "dash": "dot",
            },
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=history[
                "fused_estimate_x"
            ],
            y=history[
                "fused_estimate_y"
            ],
            mode="lines+markers",
            name="融合估计",
            line={
                "color": "#16A34A",
                "width": 3,
            },
            marker={
                "size": 4,
            },
        )
    )

    trajectory_figure.update_layout(
        height=500,
        margin={
            "l": 10,
            "r": 10,
            "t": 10,
            "b": 10,
        },
        xaxis_title="X位置 / m",
        yaxis_title="Y位置 / m",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hovermode="closest",
    )

    trajectory_figure.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    st.plotly_chart(
        trajectory_figure,
        use_container_width=True,
    )


with right_column:
    st.subheader("当前帧信息")

    status_parts = []

    if bool(current_frame["occluded"]):
        status_parts.append("位于遮挡区域")
    else:
        status_parts.append("非遮挡区域")

    if bool(
        current_frame["vehicle_available"]
    ):
        status_parts.append("车端有效")
    else:
        status_parts.append("车端漏检")

    if bool(
        current_frame["roadside_available"]
    ):
        status_parts.append("路侧有效")
    else:
        status_parts.append("路侧丢包")

    status_text = "｜".join(status_parts)

    st.info(
        f"t = {current_frame['time']:.1f} s"
        f"｜目标ID 101｜{status_text}"
    )

    st.write("**当前处理流程**")

    st.success(
        current_event["event"]
    )

    roadside_weight = float(
        current_frame["roadside_weight"]
    )

    st.progress(
        roadside_weight,
        text=(
            "当前路侧融合权重："
            f"{roadside_weight:.4f}"
        ),
    )

    association_distance = current_frame[
        "association_distance"
    ]

    if association_distance == association_distance:
        association_text = (
            f"{association_distance:.4f} m"
        )
    else:
        association_text = "无双端关联"

    current_information = {
        "字段": [
            "目标类别",
            "真实位置",
            "单车估计位置",
            "融合估计位置",
            "关联距离",
            "时间补偿",
        ],
        "值": [
            "vehicle",
            (
                f"({current_frame['truth_x']:.2f}, "
                f"{current_frame['truth_y']:.2f}) m"
            ),
            (
                f"({current_frame['vehicle_estimate_x']:.2f}, "
                f"{current_frame['vehicle_estimate_y']:.2f}) m"
            ),
            (
                f"({current_frame['fused_estimate_x']:.2f}, "
                f"{current_frame['fused_estimate_y']:.2f}) m"
            ),
            association_text,
            f"{delay_ms} ms",
        ],
    }

    st.dataframe(
        current_information,
        hide_index=True,
        use_container_width=True,
    )


st.divider()
st.subheader("融合权重变化")

weight_figure = go.Figure()

weight_figure.add_trace(
    go.Scatter(
        x=frames["time"],
        y=frames["vehicle_weight"],
        mode="lines",
        name="车端权重",
        line={
            "color": "#2563EB",
            "width": 2,
        },
    )
)

weight_figure.add_trace(
    go.Scatter(
        x=frames["time"],
        y=frames["roadside_weight"],
        mode="lines",
        name="路侧权重",
        line={
            "color": "#F59E0B",
            "width": 2,
        },
    )
)

weight_figure.add_vrect(
    x0=4.0,
    x1=7.0,
    fillcolor="#DC2626",
    opacity=0.08,
    line_width=0,
    annotation_text="车端遮挡区间",
    annotation_position="top left",
)

weight_figure.update_layout(
    height=330,
    margin={
        "l": 10,
        "r": 10,
        "t": 30,
        "b": 10,
    },
    xaxis_title="时间 / s",
    yaxis_title="融合权重",
    yaxis_range=[0.0, 1.0],
    legend={
        "orientation": "h",
    },
)

st.plotly_chart(
    weight_figure,
    use_container_width=True,
)


st.subheader("固定权重与自适应权重消融")

adaptive_result = run_simulation(
    delay_s=delay_ms / 1000.0,
    loss_rate=loss_percent / 100.0,
    occlusion=occlusion,
    adaptive=True,
    seed=int(seed),
)

fixed_result = run_simulation(
    delay_s=delay_ms / 1000.0,
    loss_rate=loss_percent / 100.0,
    occlusion=occlusion,
    adaptive=False,
    seed=int(seed),
)

comparison_data = {
    "融合策略": [
        "固定权重",
        "自适应权重",
    ],
    "融合RMSE / m": [
        round(
            fixed_result.metrics[
                "fused_rmse"
            ],
            4,
        ),
        round(
            adaptive_result.metrics[
                "fused_rmse"
            ],
            4,
        ),
    ],
    "相对单车改善 / %": [
        round(
            fixed_result.metrics[
                "improvement_pct"
            ],
            2,
        ),
        round(
            adaptive_result.metrics[
                "improvement_pct"
            ],
            2,
        ),
    ],
    "双端帧平均路侧权重": [
        round(
            fixed_result.metrics[
                "mean_paired_roadside_weight"
            ],
            4,
        ),
        round(
            adaptive_result.metrics[
                "mean_paired_roadside_weight"
            ],
            4,
        ),
    ],
}

st.dataframe(
    comparison_data,
    hide_index=True,
    use_container_width=True,
)


