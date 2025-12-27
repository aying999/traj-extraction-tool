import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # 引入 GO 库用于图层合并
import os

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(layout="wide", page_title="Waymo 轨迹可视化 Pro")
st.title("🚗 Waymo Motion Dataset 终极可视化工具")

# CSS 微调：减小顶部空白
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据加载与缓存
# ==========================================
DATA_PATH = "output/data_nuscenes.csv"

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    # 读取 CSV
    df = pd.read_csv(file_path)
    return df

with st.spinner('正在加载海量数据，请稍候...'):
    df = load_data(DATA_PATH)

if df is None:
    st.error(f"❌ 找不到文件: {DATA_PATH}。请确认你已经运行了 'python extract.py' 且 output 文件夹下有 data.csv")
    st.stop()

# ==========================================
# 3. 侧边栏与交互
# ==========================================
st.sidebar.header("🕹️ 控制台")
st.sidebar.info(f"数据总帧数: {len(df)}")

# 获取所有场景 ID 并让用户选择
all_scenarios = df['scenario_id'].unique()
selected_scenario = st.sidebar.selectbox("选择场景 ID (Scenario ID)", all_scenarios)

# ==========================================
# 4. 数据预处理 (针对选中场景)
# ==========================================
scene_data = df[df['scenario_id'] == selected_scenario].copy()

# 【自动修复】防止缺少 frame_id
if 'frame_id' not in scene_data.columns:
    # st.sidebar.warning("⚠️ 正在自动生成 frame_id...")
    times = sorted(scene_data['timestamp'].unique())
    time_map = {t: i for i, t in enumerate(times)}
    scene_data['frame_id'] = scene_data['timestamp'].map(time_map)

# 必须按帧排序，否则动画会跳变
scene_data = scene_data.sort_values(by="frame_id")

# 【计算速度】用于着色 (km/h)
if 'vx' in scene_data.columns and 'vy' in scene_data.columns:
    scene_data['speed_kmh'] = (scene_data['vx']**2 + scene_data['vy']**2)**0.5 * 3.6
else:
    scene_data['speed_kmh'] = 0

# 分离自车和其他车
ego_data = scene_data[scene_data['is_ego'] == True]
others_data = scene_data[scene_data['is_ego'] == False]

# ==========================================
# 5. 可视化核心逻辑 (分层绘制)
# ==========================================
st.subheader(f"📍 场景预览: {selected_scenario}")

# --- 第一层：绘制背景轨迹 (静态) ---
# 画其他车辆的轨迹 (极淡的白色幽灵线)
fig_bg = px.line(
    others_data,
    x="x", y="y",
    line_group="track_id",
    color_discrete_sequence=["rgba(255, 255, 255, 0.15)"], 
)
fig_bg.update_traces(line=dict(width=1), hoverinfo="skip") # 禁用背景线悬停

# 画自车(Ego)的轨迹 (醒目红色虚线)
if not ego_data.empty:
    ego_trace = px.line(
        ego_data, x="x", y="y", line_group="track_id",
        color_discrete_sequence=["rgba(255, 50, 50, 0.9)"] 
    ).data[0]
    ego_trace.line.width = 4
    ego_trace.line.dash = 'dash' 
    fig_bg.add_trace(ego_trace)

# --- 第二层：绘制动态物体 (动画层) ---
# 颜色代表速度，使用 Plasma 配色
fig_ani = px.scatter(
    scene_data, 
    x="x", y="y", 
    color="speed_kmh",     
    range_color=[0, 80],   # 速度范围 0-80 km/h
    color_continuous_scale="Plasma", 
    
    animation_frame="frame_id", 
    animation_group="track_id",
    
    size="length",         # 大小映射车长
    size_max=12,
    opacity=0.9,
    
    hover_data=["speed_kmh", "type", "track_id"],
    title="灰色=背景轨迹 | 红色虚线=自车规划 | 彩色点=实时位置"
)

# --- 第三层：图层合并 (使用 go.Figure 修复报错) ---
# 提取所有数据：背景线在前(底层)，动画点在后(顶层)
final_data = list(fig_bg.data) + list(fig_ani.data)

# 创建新图表对象
fig_final = go.Figure(
    data=final_data,
    layout=fig_ani.layout,
    frames=fig_ani.frames
)

# --- 第四层：视觉美化与比例修正 ---
fig_final.update_layout(
    plot_bgcolor='black',      # 绘图区纯黑
    paper_bgcolor='#0e1117',   # 网页背景深色
    
    # ✨✨✨ 核心修复：锁定比例，防止变扁 ✨✨✨
    xaxis=dict(showgrid=False, visible=False), 
    yaxis=dict(
        showgrid=False, 
        visible=False, 
        scaleanchor="x", # 锚定 X 轴
        scaleratio=1     # 比例 1:1
    ), 
    
    font=dict(color="white"),  
    coloraxis_colorbar=dict(title="速度 km/h"),
    height=800, # 增加高度，适应锁定比例后的留白
    margin=dict(l=0, r=0, t=40, b=0),
    
    # 调整播放按钮位置
    updatemenus=[dict(type='buttons', showactive=False, x=0.1, y=0, xanchor='right', yanchor='top')]
)

# ==========================================
# 6. 渲染输出
# ==========================================
st.plotly_chart(fig_final, use_container_width=True)

# 底部数据展示
with st.expander("🔍 查看该场景的原始数据"):
    st.dataframe(scene_data)