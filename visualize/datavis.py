import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(layout="wide", page_title="Waymo HD Map Visualizer Pro", page_icon="🚘")
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    h1 {color: #00f2ff; font-family: sans-serif;}
    div[data-testid="stMetricValue"] {color: #ff0055;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 工具函数：计算车辆矩形框坐标
# ==========================================
def get_vehicle_box_coords(row):
    """
    根据车辆中心点、朝向、长宽，计算旋转后的四个角坐标。
    返回用于 Plotly 绘制多边形的封闭坐标数组 (5个点，回到起点)。
    """
    # 尝试获取长宽，如果没有则使用默认标准轿车尺寸
    L = row.get('length', 4.7) 
    W = row.get('width', 2.0)
    
    cx, cy = row['x'], row['y']
    theta = row['heading'] # 假设这里是弧度制

    # 矩形四个角相对于中心的未旋转坐标
    # 顺序：后右 -> 后左 -> 前左 -> 前右 (逆时针)
    l2, w2 = L / 2.0, W / 2.0
    corners_rel = np.array([
        [-l2, -w2],
        [-l2,  w2],
        [ l2,  w2],
        [ l2, -w2]
    ])

    # 旋转矩阵
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])

    # 旋转并平移
    corners_rotated = corners_rel.dot(rot_matrix.T) 
    corners_final = corners_rotated + np.array([cx, cy])

    # 闭合多边形
    x_coords = np.append(corners_final[:, 0], corners_final[0, 0])
    y_coords = np.append(corners_final[:, 1], corners_final[0, 1])

    return x_coords, y_coords

# ==========================================
# 3. 数据加载
# ==========================================
TRAJ_PATH = "output/data_waymo.csv"
MAP_PATH = "output/map_waymo.csv"

@st.cache_data
def load_data(traj_path, map_path):
    if not os.path.exists(traj_path):
        return None, None
    df_traj = pd.read_csv(traj_path)
    df_map = None
    if os.path.exists(map_path):
        df_map = pd.read_csv(map_path)
    return df_traj, df_map

with st.spinner('🚀 正在加载 Waymo 数据...'):
    df_traj, df_map = load_data(TRAJ_PATH, MAP_PATH)

if df_traj is None:
    st.error("❌ 找不到轨迹文件。请检查 output/data_waymo.csv 是否存在。")
    st.stop()

# ==========================================
# 4. 交互控制与预处理
# ==========================================
st.sidebar.title("🎛️ 控制中心")
all_scenarios = df_traj['scenario_id'].unique()
selected_scenario = st.sidebar.selectbox("📍 选择场景 (Scenario ID)", all_scenarios)

# 过滤数据
scene_traj = df_traj[df_traj['scenario_id'] == selected_scenario].copy()
scene_map = df_map[df_map['scenario_id'] == selected_scenario].copy() if df_map is not None else pd.DataFrame()

# 生成 Frame ID
if 'frame_id' not in scene_traj.columns:
    times = sorted(scene_traj['timestamp'].unique())
    time_map = {t: i for i, t in enumerate(times)}
    scene_traj['frame_id'] = scene_traj['timestamp'].map(time_map)
scene_traj = scene_traj.sort_values(by="frame_id")

# 计算速度
if 'vx' in scene_traj.columns:
    scene_traj['speed_kmh'] = (scene_traj['vx']**2 + scene_traj['vy']**2)**0.5 * 3.6
else:
    scene_traj['speed_kmh'] = 0

# ==========================================
# 5. 核心绘图 (HD Map + 矩形车辆动画)
# ==========================================
st.title(f"🛣️ Waymo 场景回放: 真实车辆模型视图")

# 初始化 Figure
fig = go.Figure()

# --- [Layer 1] 静态地图绘制 ---
# 注意：这些 Trace 会先被加入 fig.data，索引从 0 开始
if not scene_map.empty:
    # 道路边缘
    edges = scene_map[scene_map['type'] == 'ROAD_EDGE']
    for fid, group in edges.groupby('feature_id'):
        group = group.sort_values('order')
        fig.add_trace(go.Scatter(
            x=group['x'], y=group['y'],
            mode='lines', line=dict(color='#F4D03F', width=2), 
            hoverinfo='skip', showlegend=False
        ))
    
    # 道路标线
    lines = scene_map[scene_map['type'] == 'ROAD_LINE']
    for fid, group in lines.groupby('feature_id'):
        group = group.sort_values('order')
        fig.add_trace(go.Scatter(
            x=group['x'], y=group['y'],
            mode='lines', line=dict(color='rgba(200, 200, 200, 0.4)', width=1, dash='dash'), 
            hoverinfo='skip', showlegend=False
        ))

# --- [关键修复] 获取动态图层的正确索引 ---
# 在添加车辆 Trace 之前，计算当前已经有多少个地图 Trace
# 这确保了动画帧只更新车辆所在的那个 Trace，而不是错误地去更新地图线条
vehicle_trace_index = len(fig.data)

# --- [Layer 2] 准备动画数据 ---
sorted_frame_ids = sorted(scene_traj['frame_id'].unique())
frames = []

# 用于存储第一帧数据（作为底图初始状态）
init_x = []
init_y = []
init_hover = []

# 遍历每一帧构建多边形数据
for fid in sorted_frame_ids:
    frame_data = scene_traj[scene_traj['frame_id'] == fid]
    
    # 当前帧所有车辆的坐标列表（用 None 分隔）
    frame_x = []
    frame_y = []
    frame_hover = []
    
    for _, row in frame_data.iterrows():
        # 仅处理车辆，如果是行人可以用简单的点处理（此处略过行人以简化逻辑）
        if 'VEHICLE' in str(row['type']).upper():
            xs, ys = get_vehicle_box_coords(row)
            
            frame_x.extend(xs)
            frame_x.append(None) # Plotly技巧：用None断开不同图形
            
            frame_y.extend(ys)
            frame_y.append(None)
            
            # 悬停信息也需要对应点的数量
            info = f"ID: {row['track_id']}<br>Speed: {row['speed_kmh']:.1f} km/h"
            frame_hover.extend([info] * len(xs))
            frame_hover.append(None)
            
    # 如果是第一帧，保存给初始 Trace 使用
    if fid == sorted_frame_ids[0]:
        init_x = frame_x
        init_y = frame_y
        init_hover = frame_hover
        
    # 创建动画帧
    frames.append(go.Frame(
        data=[go.Scatter(
            x=frame_x,
            y=frame_y,
            hovertext=frame_hover
        )],
        # [关键] 显式指定这帧数据属于哪个 Trace ID
        traces=[vehicle_trace_index], 
        name=str(fid)
    ))

# --- [Layer 2] 添加车辆初始 Trace ---
fig.add_trace(go.Scatter(
    x=init_x,
    y=init_y,
    mode='lines', # 显示轮廓
    fill='toself', # 内部填充
    fillcolor='rgba(0, 242, 255, 0.7)', # 青色半透明
    line=dict(color='#ffffff', width=1), # 白色边框
    hoverinfo='text',
    hovertext=init_hover,
    name='Vehicles',
    showlegend=False
))

# 注入 Frames
fig.frames = frames

# --- 视觉与布局设置 ---
fig.update_layout(
    plot_bgcolor='#000000',
    paper_bgcolor='#0e1117',
    # 强制等比例显示，防止车身变形
    xaxis=dict(visible=False, showgrid=False, scaleanchor="y", scaleratio=1),
    yaxis=dict(visible=False, showgrid=False),
    font=dict(color="#a0a0a0"),
    height=800,
    margin=dict(t=40, b=0, l=0, r=0),
    
    # 播放按钮配置
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        y=1, x=0.1, xanchor='right', yanchor='top',
        pad=dict(t=0, r=10),
        buttons=[dict(label='▶ Play',
                      method='animate',
                      args=[None, dict(frame=dict(duration=100, redraw=True), # redraw=True 对多边形动画很重要
                                       fromcurrent=True,
                                       mode='immediate')])]
    )]
)

st.plotly_chart(fig, use_container_width=True)

# 底部统计栏
col1, col2, col3 = st.columns(3)
veh_count = scene_traj[scene_traj['type'].str.contains('VEHICLE', na=False)]['track_id'].nunique()
col1.metric("🚗 动态车辆数", veh_count)
col2.metric("🛣️ 地图元素数", len(scene_map))
col3.metric("⏱️ 动画总帧数", len(sorted_frame_ids))