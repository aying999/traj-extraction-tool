import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_and_process_data(traj_path, map_path, scenario_id):
    """
    加载并处理场景数据，执行动静分离逻辑
    """
    if not os.path.exists(traj_path): 
        return None, None, None, None, None
    
    # 1. 读取数据
    df_traj = pd.read_csv(traj_path)
    df_map = pd.read_csv(map_path) if os.path.exists(map_path) else pd.DataFrame()
    
    # 2. 过滤场景
    scene_traj = df_traj[df_traj['scenario_id'] == scenario_id].copy()
    scene_map = df_map[df_map['scenario_id'] == scenario_id].copy() if not df_map.empty else pd.DataFrame()

    # 3. 生成 Frame ID
    if 'frame_id' not in scene_traj.columns:
        times = sorted(scene_traj['timestamp'].unique())
        time_map = {t: i for i, t in enumerate(times)}
        scene_traj['frame_id'] = scene_traj['timestamp'].map(time_map)
    scene_traj = scene_traj.sort_values(by="frame_id")

    # 4. 计算速度 (km/h)
    if 'vx' in scene_traj.columns:
        scene_traj['speed_kmh'] = (scene_traj['vx']**2 + scene_traj['vy']**2)**0.5 * 3.6
    else:
        scene_traj['speed_kmh'] = 0

    # 5. 核心分类逻辑 (动静分离 & VRU提取)
    track_stats = scene_traj.groupby('track_id').agg({
        'speed_kmh': 'max',
        'type': 'first'
    })
    
    is_vehicle = track_stats['type'].astype(str).str.contains('VEHICLE')
    is_low_speed = track_stats['speed_kmh'] < 1.0
    
    # 静态物体：是车 且 速度极低
    static_mask = is_vehicle & is_low_speed
    
    static_track_ids = track_stats[static_mask].index.tolist()
    active_track_ids = track_stats[~static_mask].index.tolist()
    
    # 数据集切分
    static_df = scene_traj[scene_traj['track_id'].isin(static_track_ids)]
    active_df = scene_traj[scene_traj['track_id'].isin(active_track_ids)]
    
    # 活跃物体细分
    moving_cars_df = active_df[active_df['type'].str.contains('VEHICLE', na=False)]
    vrus_df = active_df[~active_df['type'].str.contains('VEHICLE', na=False)]

    # 静态物体只保留首帧用于绘图
    static_df_first = static_df.groupby('track_id').first().reset_index()

    return scene_traj, scene_map, static_df_first, moving_cars_df, vrus_df

def get_all_scenarios(traj_path):
    """获取所有场景ID列表"""
    if not os.path.exists(traj_path):
        return []
    df = pd.read_csv(traj_path, usecols=['scenario_id'])
    return df['scenario_id'].unique()