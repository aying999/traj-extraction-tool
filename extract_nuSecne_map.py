import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import os
import numpy as np

# --- 配置 ---
DATAROOT = "./nuscenes_data" # 你的数据路径
VERSION = "v1.0-mini"
OUTPUT_MAP_FILE = "output/map_nuscenes.csv"

def extract_maps():
    print(f"⏳ 正在加载 nuScenes ({VERSION})...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    
    # 缓存已加载的地图 API (避免重复加载)
    maps = {}
    
    all_map_features = []
    
    print("🗺️ 开始提取场景地图数据...")
    for scene in tqdm(nusc.scene):
        scene_id = scene['name']
        log = nusc.get('log', scene['log_token'])
        location = log['location'] # e.g., 'singapore-onenorth'
        
        # 1. 动态加载该城市的地图
        if location not in maps:
            print(f"   >>> Loading map: {location}...")
            maps[location] = NuScenesMap(dataroot=DATAROOT, map_name=location)
        nusc_map = maps[location]
        
        # 2. 计算该场景的物理范围 (Bounding Box)
        # 我们遍历场景里所有的 Sample，找到最大最小的 x, y
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        
        x_coords = []
        y_coords = []
        
        # 为了速度，我们只采样头、中、尾几帧来确定范围
        # (严谨做法是遍历所有帧，但对于提取地图范围，采样足够了)
        samples_to_check = []
        curr = sample_token
        while curr:
            samples_to_check.append(curr)
            curr = nusc.get('sample', curr)['next']
            
        # 稀疏采样 (每隔 5 帧采一个点)
        for token in samples_to_check[::5]:
            sample = nusc.get('sample', token)
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            x_coords.append(ego_pose['translation'][0])
            y_coords.append(ego_pose['translation'][1])
            
        x_min, x_max = min(x_coords) - 50, max(x_coords) + 50 # 向外扩 50米 保证视野
        y_min, y_max = min(y_coords) - 50, max(y_coords) + 50
        
        # 3. 提取地图图层
        # 我们主要关心：车道线、路沿、人行道
        layer_names = ['lane', 'road_segment', 'ped_crossing', 'road_divider', 'lane_divider']
        
        # 获取范围内的所有地图记录
        records = nusc_map.get_records_in_patch((x_min, y_min, x_max, y_max), layer_names, mode='intersect')
        
        # 4. 将记录转换为坐标点
        for layer_name in layer_names:
            for token in records[layer_name]:
                # 获取几何形状 (Polygon 或 LineString)
                # nuScenes API 稍微有点绕，需要根据 token 获取 record，再获取 nodes
                
                try:
                    # 获取该元素的 token 和详细信息
                    record = nusc_map.get(layer_name, token)
                    
                    if layer_name == 'lane':
                        # 对于车道，我们提取“中心线” (Centerline)
                        # 注意：nusc_map.get_arcline_path 是获取拟合曲线，这里我们直接取离散化节点
                        # 简单的做法：提取 polygon 的边界或者利用 connectivity
                        # 更简单的可视化做法：提取 lane_divider (车道分隔线)
                        pass 
                    
                    # 为了可视化简单，我们重点提取 'road_divider' 和 'lane_divider' (白虚线/实线)
                    # 以及 'road_edge' (马路牙子 - 注意 mini 版可能有不同命名，通常用 road_segment 的边界)
                    
                    # 让我们用更通用的方法：获取 geometry token
                    # 但 NuScenesMap API 的 get_discretized_nodes 更方便
                    pass
                except:
                    continue

        # --- 简化策略：直接提取线条 ---
        # 上面的通用逻辑太复杂，我们用 render 逻辑反推
        # 直接利用 nusc_map 自带的提取 line 的功能
        
        patch_box = (x_min, y_min, x_max, y_max)
        patch_angle = 0 
        
        # 提取车道分隔线 (虚线/实线)
        for layer in ['lane_divider', 'road_divider']:
             for token in records[layer]:
                record = nusc_map.get(layer, token)
                line = nusc_map.extract_line(record['line_token'])
                # line 是一个 list of nodes [(x,y), (x,y)...]
                
                # 存入列表
                for i, (x, y) in enumerate(line):
                    all_map_features.append({
                        'scenario_id': scene_id,
                        'line_id': token, # 用 token 区分不同的线
                        'type': 'LANE_LINE', # 统一叫车道线
                        'x': x, 'y': y,
                        'order': i # 记录点的顺序，画线用
                    })

        # 提取人行道 (作为多边形轮廓)
        for token in records['ped_crossing']:
            record = nusc_map.get('ped_crossing', token)
            # 获取外部轮廓节点
            nodes = nusc_map.get('node', record['exterior_node_tokens'])
            for i, node in enumerate(nodes):
                all_map_features.append({
                    'scenario_id': scene_id,
                    'line_id': token,
                    'type': 'CROSSWALK',
                    'x': node['x'], 'y': node['y'],
                    'order': i
                })
                
    # 保存
    if not os.path.exists("output"):
        os.makedirs("output")
        
    df = pd.DataFrame(all_map_features)
    df.to_csv(OUTPUT_MAP_FILE, index=False)
    print(f"✅ 地图提取完成！保存到: {OUTPUT_MAP_FILE}")

if __name__ == "__main__":
    extract_maps()