import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import os
import numpy as np


DATAROOT = "./nuscenes_data" 
VERSION = "v1.0-mini"
OUTPUT_MAP_FILE = "output/map_nuscenes.csv"

def extract_maps():
    print(f"⏳ 正在加载 nuScenes ({VERSION})...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    
   
    maps = {}
    
    all_map_features = []
    
    print("🗺️ 开始提取场景地图数据...")
    for scene in tqdm(nusc.scene):
        scene_id = scene['name']
        log = nusc.get('log', scene['log_token'])
        location = log['location'] # e.g., 'singapore-onenorth'
        
        
        if location not in maps:
            print(f"   >>> Loading map: {location}...")
            maps[location] = NuScenesMap(dataroot=DATAROOT, map_name=location)
        nusc_map = maps[location]
        
        
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        
        x_coords = []
        y_coords = []
        
        
        samples_to_check = []
        curr = sample_token
        while curr:
            samples_to_check.append(curr)
            curr = nusc.get('sample', curr)['next']
            
     
        for token in samples_to_check[::5]:
            sample = nusc.get('sample', token)
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            x_coords.append(ego_pose['translation'][0])
            y_coords.append(ego_pose['translation'][1])
            
        x_min, x_max = min(x_coords) - 50, max(x_coords) + 50 # 向外扩 50米 保证视野
        y_min, y_max = min(y_coords) - 50, max(y_coords) + 50
        
        
        layer_names = ['lane', 'road_segment', 'ped_crossing', 'road_divider', 'lane_divider']
        
        
        records = nusc_map.get_records_in_patch((x_min, y_min, x_max, y_max), layer_names, mode='intersect')
        
       
        for layer_name in layer_names:
            for token in records[layer_name]:
                
                try:
                  
                    record = nusc_map.get(layer_name, token)
                    
                    if layer_name == 'lane':
                        pass 
                    
                
                    pass
                except:
                    continue

        
        
        patch_box = (x_min, y_min, x_max, y_max)
        patch_angle = 0 
        
        
        for layer in ['lane_divider', 'road_divider']:
             for token in records[layer]:
                record = nusc_map.get(layer, token)
                line = nusc_map.extract_line(record['line_token'])
                
                for i, (x, y) in enumerate(line):
                    all_map_features.append({
                        'scenario_id': scene_id,
                        'line_id': token,
                        'type': 'LANE_LINE', 
                        'x': x, 'y': y,
                        'order': i 
                    })

  
        for token in records['ped_crossing']:
            record = nusc_map.get('ped_crossing', token)
          
            nodes = nusc_map.get('node', record['exterior_node_tokens'])
            for i, node in enumerate(nodes):
                all_map_features.append({
                    'scenario_id': scene_id,
                    'line_id': token,
                    'type': 'CROSSWALK',
                    'x': node['x'], 'y': node['y'],
                    'order': i
                })
                

    if not os.path.exists("output"):
        os.makedirs("output")
        
    df = pd.DataFrame(all_map_features)
    df.to_csv(OUTPUT_MAP_FILE, index=False)
    print(f"✅ 地图提取完成！保存到: {OUTPUT_MAP_FILE}")

if __name__ == "__main__":
    extract_maps()