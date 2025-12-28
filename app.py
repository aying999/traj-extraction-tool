import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils import load_config, get_box_coords
from data_processor import load_and_process_data, get_all_scenarios

cfg = load_config()

st.set_page_config(
    layout=cfg['app']['layout'], 
    page_title=cfg['app']['title'], 
    page_icon=cfg['app']['icon']
)


st.markdown(f"""
<style>
    .stApp {{background-color: {cfg['visuals']['background_color']};}}
    h1 {{color: #00f2ff; font-family: sans-serif;}}
    div[data-testid="stMetricValue"] {{color: #ff0055;}}
    div[data-testid="stDataFrame"] {{width: 100%;}}
</style>
""", unsafe_allow_html=True)


st.sidebar.title("🎛️ 控制中心")

traj_path = cfg['paths']['traj_file']
map_path = cfg['paths']['map_file']


all_scenarios = get_all_scenarios(traj_path)
if len(all_scenarios) == 0:
    st.error(f"❌ 找不到轨迹文件: {traj_path}")
    st.stop()

selected_scenario = st.sidebar.selectbox("📍 选择场景 (Scenario ID)", all_scenarios)

with st.spinner('🚀 正在解析全量交通参与者...'):
    scene_traj, scene_map, static_df, moving_cars_df, vrus_df = \
        load_and_process_data(traj_path, map_path, selected_scenario)

sorted_frame_ids = sorted(scene_traj['frame_id'].unique())


st.title(f"🚘 {cfg['app']['title']}")
fig = go.Figure()


if not scene_map.empty:
    for fid, group in scene_map[scene_map['type'] == 'ROAD_EDGE'].groupby('feature_id'):
        fig.add_trace(go.Scatter(
            x=group.sort_values('order')['x'], y=group.sort_values('order')['y'],
            mode='lines', line=dict(color=cfg['visuals']['map']['road_edge'], width=2), hoverinfo='skip'))
    for fid, group in scene_map[scene_map['type'] == 'ROAD_LINE'].groupby('feature_id'):
        fig.add_trace(go.Scatter(
            x=group.sort_values('order')['x'], y=group.sort_values('order')['y'],
            mode='lines', line=dict(color=cfg['visuals']['map']['road_line'], width=1, dash='dash'), hoverinfo='skip'))


static_x, static_y, static_hover = [], [], []
for _, row in static_df.iterrows():
    xs, ys = get_box_coords(row, cfg)
    static_x.extend(xs); static_x.append(None)
    static_y.extend(ys); static_y.append(None)
    static_hover.extend([f"Static<br>ID: {row['track_id']}"]*5); static_hover.append(None)

fig.add_trace(go.Scatter(
    x=static_x, y=static_y, mode='lines', fill='toself',
    fillcolor=cfg['visuals']['vehicles']['static_color'], 
    line=dict(color=cfg['visuals']['vehicles']['static_border'], width=1),
    hoverinfo='text', hovertext=static_hover, name='Static Vehicles'
))


trail_x, trail_y = [], []
all_active = pd.concat([moving_cars_df, vrus_df])
for tid, group in all_active.groupby('track_id'):
    trail_x.extend(group['x'].tolist()); trail_x.append(None)
    trail_y.extend(group['y'].tolist()); trail_y.append(None)

fig.add_trace(go.Scatter(
    x=trail_x, y=trail_y, mode='lines',
    line=dict(color=cfg['visuals']['trail']['color'], width=1), 
    hoverinfo='skip', name='Trails'
))


f0_cars = moving_cars_df[moving_cars_df['frame_id'] == sorted_frame_ids[0]]
cx, cy = [], []
for _, row in f0_cars.iterrows():
    xs, ys = get_box_coords(row, cfg)
    cx.extend(xs); cx.append(None); cy.extend(ys); cy.append(None)
fig.add_trace(go.Scatter(
    x=cx, y=cy, mode='lines', fill='toself', 
    fillcolor=cfg['visuals']['vehicles']['moving_color'], 
    line=dict(color='white', width=1), name='Moving Cars'))


f0_vrus = vrus_df[vrus_df['frame_id'] == sorted_frame_ids[0]]
vx, vy = [], []
for _, row in f0_vrus.iterrows():
    xs, ys = get_box_coords(row, cfg)
    vx.extend(xs); vx.append(None); vy.extend(ys); vy.append(None)
fig.add_trace(go.Scatter(
    x=vx, y=vy, mode='lines', fill='toself', 
    fillcolor=cfg['visuals']['vrus']['color'], 
    line=dict(color='white', width=1), name='Pedestrians/Cyclists'))


frames = []
for fid in sorted_frame_ids:
    
    f_cars = moving_cars_df[moving_cars_df['frame_id'] == fid]
    car_x, car_y, car_h = [], [], []
    for _, row in f_cars.iterrows():
        xs, ys = get_box_coords(row, cfg)
        car_x.extend(xs); car_x.append(None); car_y.extend(ys); car_y.append(None)
        car_h.extend([f"Car<br>ID: {row['track_id']}<br>V: {row['speed_kmh']:.1f}"]*5); car_h.append(None)
        
    
    f_vrus = vrus_df[vrus_df['frame_id'] == fid]
    vru_x, vru_y, vru_h = [], [], []
    for _, row in f_vrus.iterrows():
        xs, ys = get_box_coords(row, cfg)
        vru_x.extend(xs); vru_x.append(None); vru_y.extend(ys); vru_y.append(None)
        vru_h.extend([f"{row['type']}<br>ID: {row['track_id']}"]*5); vru_h.append(None)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=car_x, y=car_y, hovertext=car_h),
            go.Scatter(x=vru_x, y=vru_y, hovertext=vru_h)
        ],
        name=str(fid),
        traces=[len(fig.data)-2, len(fig.data)-1]
    ))

fig.frames = frames

fig.update_layout(
    plot_bgcolor=cfg['visuals']['plot_bgcolor'], 
    paper_bgcolor=cfg['visuals']['background_color'],
    xaxis=dict(visible=False, showgrid=False, scaleanchor="y", scaleratio=1),
    yaxis=dict(visible=False, showgrid=False),
    font=dict(color="#a0a0a0"), height=800, margin=dict(t=40, b=0, l=0, r=0),
    updatemenus=[dict(type='buttons', showactive=False, y=1, x=0.1, xanchor='right', yanchor='top', pad=dict(t=0, r=10),
                      buttons=[dict(label='▶ Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True, mode='immediate')])])]
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("### 📊 场景全量统计")

duration = len(sorted_frame_ids) * 0.1
map_w = scene_traj['x'].max() - scene_traj['x'].min()
map_h = scene_traj['y'].max() - scene_traj['y'].min()

n_moving_cars = moving_cars_df['track_id'].nunique()
n_static = len(static_df)
n_ped = vrus_df[vrus_df['type'].str.contains('PEDESTRIAN')]['track_id'].nunique()
n_cyc = vrus_df[vrus_df['type'].str.contains('CYCLIST')]['track_id'].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("⏱️ 场景时长", f"{duration:.1f} s")
k2.metric("🚀 最高车速", f"{scene_traj['speed_kmh'].max():.1f} km/h")
k3.metric("📏 场景范围", f"{int(map_w)}m x {int(map_h)}m")
k4.metric("👥 总参与者", f"{scene_traj['track_id'].nunique()}")

st.caption("🚦 动静与类别构成")
s1, s2, s3, s4 = st.columns(4)
s1.metric("🚗 移动车辆 (Moving)", f"{n_moving_cars}", delta="黄色高亮")
s2.metric("🅿️ 静止车辆 (Static)", f"{n_static}", delta="灰色背景", delta_color="off")
s3.metric("🚶 行人 (Pedestrian)", f"{n_ped}", delta="橙色高亮", delta_color="inverse")
s4.metric("🚲 骑行者 (Cyclist)", f"{n_cyc}", delta="橙色高亮", delta_color="inverse")


st.markdown("---")
st.subheader("📋 原始数据详情")
with st.expander("🔍 展开数据表", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1: 
        types = ['ALL'] + sorted(list(scene_traj['type'].astype(str).unique()))
        sel_type = st.selectbox("筛选类型", types)
    with f2:
        if sel_type!='ALL': df_f = scene_traj[scene_traj['type']==sel_type]
        else: df_f = scene_traj
        ids = ['ALL'] + sorted(list(df_f['track_id'].unique()))
        sel_id = st.selectbox("筛选ID", ids)
    with f3:
        sel_frame = st.select_slider("筛选帧", options=['ALL']+sorted_frame_ids)
    
    df_show = scene_traj.copy()
    if sel_type!='ALL': df_show = df_show[df_show['type']==sel_type]
    if sel_id!='ALL': df_show = df_show[df_show['track_id']==sel_id]
    if sel_frame!='ALL': df_show = df_show[df_show['frame_id']==sel_frame]
    
    st.dataframe(df_show, use_container_width=True, height=400, hide_index=True)
    csv = df_show.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 下载当前筛选数据 (CSV)", data=csv, file_name=f'waymo_data_{selected_scenario}.csv', mime='text/csv')