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
    L = row.get('length', 4.7) # 默认长度
    W = row.get('width', 2.0)  # 默认宽度
    
    cx, cy = row['x'], row['y']
    # 确保使用弧度制的 heading
    theta = row['heading'] 

    # 矩形四个角相对于中心的未旋转坐标
    # 顺序：后右 -> 后左 -> 前左 -> 前右 (逆时针)
    l2, w2 = L / 2.0, W / 2.0
    corners_rel = np.array([
        [-l2, -w2],
        [-l2,  w2],
        [ l2,  w2],
        [ l2, -w2]
    ])

    # 旋转矩阵 (标准二维旋转)
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])

    # 旋转并平移到实际位置
    # dot product: (4x2) dot (2x2) = 4x2
    corners_rotated = corners_rel.dot(rot_matrix.T) 
    corners_final = corners_rotated + np.array([cx, cy])

    # 为了画封闭多边形，需要在末尾重复第一个点
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

with st.spinner('🚀 正在加载 Waymo 数据，准备构建车辆模型...'):
    df_traj, df_map = load_data(TRAJ_PATH, MAP_PATH)

if df_traj is None:
    st.error("❌ 找不到轨迹文件。")
    st.stop()

# ==========================================
# 4. 交互控制与数据预处理
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

# 计算速度 (仅用于显示信息，不再用于颜色映射，因为多边形颜色映射比较复杂)
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

# --- 层 1: 静态地图 (保持不变) ---
if not scene_map.empty:
    # 道路边缘
    for fid, group in scene_map[scene_map['type'] == 'ROAD_EDGE'].groupby('feature_id'):
        fig.add_trace(go.Scatter(x=group.sort_values('order')['x'], y=group.sort_values('order')['y'],
                                 mode='lines', line=dict(color='#F4D03F', width=2), hoverinfo='skip'))
    # 道路标线
    for fid, group in scene_map[scene_map['type'] == 'ROAD_LINE'].groupby('feature_id'):
        fig.add_trace(go.Scatter(x=group.sort_values('order')['x'], y=group.sort_values('order')['y'],
                                 mode='lines', line=dict(color='rgba(200, 200, 200, 0.4)', width=1, dash='dash'), hoverinfo='skip'))

# --- 层 2: 动态车辆动画 (核心修改部分) ---

# 获取所有唯一的帧 ID，并排序
sorted_frame_ids = sorted(scene_traj['frame_id'].unique())

frames = []
# 用于收集每一帧的车辆多边形数据
# Plotly技巧：在同一个 Trace 中画多个不相连的多边形，需要在它们的坐标之间插入 None
vehicle_polygon_x = []
vehicle_polygon_y = []
hover_texts = []

# 5.1 构建每一帧的数据
for fid in sorted_frame_ids:
    frame_data = scene_traj[scene_traj['frame_id'] == fid]
    
    current_frame_x = []
    current_frame_y = []
    current_hover = []
    
    for _, row in frame_data.iterrows():
        # 仅处理车辆类型，其他类型(如行人)可以稍后用简单点表示
        if 'VEHICLE' in str(row['type']).upper():
            # 计算矩形四个角
            xs, ys = get_vehicle_box_coords(row)
            # 添加数据并在不同车辆间插入 None
            current_frame_x.extend(xs)
            current_frame_x.append(None) 
            current_frame_y.extend(ys)
            current_frame_y.append(None)
            
            # 构建悬停信息 (为了匹配 None 的结构，需要复制5次信息+1个None)
            info = f"ID: {row['track_id']}<br>Speed: {row['speed_kmh']:.1f} km/h"
            current_hover.extend([info] * 5)
            current_hover.append(None)

    # 如果是第一帧，保存作为 Figure 的初始状态数据
    if fid == sorted_frame_ids[0]:
        vehicle_polygon_x = current_frame_x
        vehicle_polygon_y = current_frame_y
        hover_texts = current_hover
        
    # 创建 Plotly 动画帧对象
    frames.append(go.Frame(
        data=[go.Scatter(
            x=current_frame_x,
            y=current_frame_y,
            hovertext=current_hover
        )],
        name=str(fid) # 帧的名称必须是字符串
    ))

# 5.2 添加初始状态的车辆 Trace 到 Figure
# 使用 fill='toself' 来填充多边形
fig.add_trace(go.Scatter(
    x=vehicle_polygon_x,
    y=vehicle_polygon_y,
    mode='lines', # 显示边框线
    fill='toself', # 填充内部颜色
    fillcolor='rgba(0, 242, 255, 0.7)', # 填充色：明亮的青色，半透明
    line=dict(color='#ffffff', width=1), # 边框色：白色细线
    hoverinfo='text',
    hovertext=hover_texts,
    name='Vehicles'
))

# 5.3 将构建好的帧序列赋值给 Figure
fig.frames = frames

# --- 视觉与动画设置 ---
fig.update_layout(
    plot_bgcolor='#000000',
    paper_bgcolor='#0e1117',
    # 强制 XY 轴等比例，保证车辆矩形不变形
    xaxis=dict(visible=False, showgrid=False, scaleanchor="y", scaleratio=1),
    yaxis=dict(visible=False, showgrid=False),
    font=dict(color="#a0a0a0"),
    height=800,
    margin=dict(t=40, b=0, l=0, r=0),
    showlegend=False,
    # 动画控制按钮
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

# 底部统计
col1, col2, col3 = st.columns(3)
col1.metric("🚗 动态车辆数", scene_traj[scene_traj['type'].str.contains('VEHICLE', na=False)]['track_id'].nunique())
col2.metric("🛣️ 地图元素数", len(scene_map))
col3.metric("⏱️ 总帧数", len(sorted_frame_ids))