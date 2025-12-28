import pandas as pd
from nuscenes.nuscenes import NuScenes
import os
from tqdm import tqdm


DATAROOT = "./nuscenes_data"  
VERSION = "v1.0-mini"
OUTPUT_FILE = "output/data_nuscenes.csv"

def extract_nuscenes():
    print(f"⏳ 正在加载 nuScenes ({VERSION})...")
    try:
        nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("💡 提示: 请确保 DATAROOT 路径正确，且该路径下有 maps, samples, v1.0-mini 等文件夹")
        return

    all_tracks = []
    
    print("🚀 开始提取轨迹并计算差分速度...")
    
    
    for scene in tqdm(nusc.scene):
        scene_id = scene['name']
        
       
        sample_token = scene['first_sample_token']
        frame_idx = 0
        
       
        while sample_token:
            sample = nusc.get('sample', sample_token)
            timestamp = sample['timestamp'] / 1e6 
            
           
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            ego_x, ego_y = ego_pose['translation'][0], ego_pose['translation'][1]
            ego_vx, ego_vy = 0, 0 
            
            
            if sample['prev']:
                prev_sample = nusc.get('sample', sample['prev'])
                prev_timestamp = prev_sample['timestamp'] / 1e6
                dt = timestamp - prev_timestamp
                
                
                prev_lidar = nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])
                prev_pose = nusc.get('ego_pose', prev_lidar['ego_pose_token'])
                
                if dt > 0:
                    ego_vx = (ego_x - prev_pose['translation'][0]) / dt
                    ego_vy = (ego_y - prev_pose['translation'][1]) / dt

            all_tracks.append({
                'scenario_id': scene_id,
                'timestamp': timestamp,
                'frame_id': frame_idx,
                'track_id': 'ego_' + scene_id[:4],
                'type': 'TYPE_VEHICLE',
                'is_ego': True,
                'x': ego_x, 'y': ego_y,
                'length': 4.5, 'width': 2.0, 'height': 1.5,
                'vx': ego_vx, 'vy': ego_vy # ✅ 现在有速度了
            })

           
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                
                
                track_id = ann['instance_token'][:8]
                category = ann['category_name']
                x, y, z = ann['translation']
                
               
                if 'vehicle' in category: obj_type = 'TYPE_VEHICLE'
                elif 'pedestrian' in category: obj_type = 'TYPE_PEDESTRIAN'
                elif 'cycle' in category: obj_type = 'TYPE_CYCLIST'
                else: obj_type = 'TYPE_OTHER'
                
               
                vx, vy = 0, 0
                
          
                if ann['prev']:
                    prev_ann = nusc.get('sample_annotation', ann['prev'])
                    
                    
                    prev_sample_token_for_ann = prev_ann['sample_token']
                    prev_sample_for_ann = nusc.get('sample', prev_sample_token_for_ann)
                    prev_time = prev_sample_for_ann['timestamp'] / 1e6
                    
                    dt = timestamp - prev_time
                    
                    if dt > 0:
                        vx = (x - prev_ann['translation'][0]) / dt
                        vy = (y - prev_ann['translation'][1]) / dt
                
                all_tracks.append({
                    'scenario_id': scene_id,
                    'timestamp': timestamp,
                    'frame_id': frame_idx,
                    'track_id': track_id,
                    'type': obj_type,
                    'is_ego': False,
                    'x': x, 'y': y,
                    'length': ann['size'][1],
                    'width': ann['size'][0],
                    'height': ann['size'][2],
                    'vx': vx, 'vy': vy # ✅ 现在有速度了
                })

            
            sample_token = sample['next']
            frame_idx += 1

    
    if not os.path.exists("output"):
        os.makedirs("output")
    
    df = pd.DataFrame(all_tracks)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 提取完成！(包含差分速度) 已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_nuscenes()