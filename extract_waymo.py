import os
import glob
import numpy as np
import pandas as pd
# 修正点 1: 改用 tfrecord_iterator
from tfrecord.reader import tfrecord_iterator
from waymo_open_dataset.protos import scenario_pb2 

# --- 配置 ---
OBJECT_TYPE_MAP = {
    0: 'TYPE_UNSET',
    1: 'TYPE_VEHICLE',
    2: 'TYPE_PEDESTRIAN',
    3: 'TYPE_CYCLIST',
    4: 'TYPE_OTHER'
}

class WaymoExtractor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_file(self, tfrecord_path):
        print(f"🚀 正在处理: {os.path.basename(tfrecord_path)}")
        all_tracks = []
        
        # 修正点 2: 直接获取原始字节流
        loader = tfrecord_iterator(tfrecord_path)
        
        count = 0
        for record in loader:
            count += 1
            try:
                # 1. 反序列化
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record)
            except Exception as e:
                print(f"⚠️ 解析第 {count} 帧时出错: {e}")
                continue
            
            scenario_id = scenario.scenario_id
            timestamps = np.array(scenario.timestamps_seconds)
            sdc_id = scenario.sdc_track_index
            
            # 2. 遍历该场景下的每个物体
            for track_idx, track in enumerate(scenario.tracks):
                track_id = track.id
                obj_type = OBJECT_TYPE_MAP.get(track.object_type, 'UNKNOWN')
                
                # 只提取车辆 (可选)
                # if obj_type != 'TYPE_VEHICLE':
                #     continue
                
                is_ego = (track_idx == sdc_id)
                
                # 3. 遍历该物体的每一帧状态
                for step_idx, state in enumerate(track.states):
                    if not state.valid:
                        continue
                        
                    all_tracks.append({
                        'scenario_id': scenario_id,
                        'timestamp': timestamps[step_idx],
                        'frame_id': step_idx,
                        'track_id': track_id,
                        'type': obj_type,
                        'is_ego': is_ego,
                        'x': state.center_x,
                        'y': state.center_y,
                        'z': state.center_z,
                        'heading': state.heading,
                        'vx': state.velocity_x,
                        'vy': state.velocity_y,
                        'length': state.length,
                        'width': state.width,
                        'height': state.height
                    })
        
        print(f"   -> 解析完成，包含 {count} 个场景")
        return pd.DataFrame(all_tracks)

    def run(self, input_path):
        if os.path.isdir(input_path):
            files = glob.glob(os.path.join(input_path, "*.tfrecord"))
        else:
            files = [input_path]
            
        if not files:
            print(f"❌ 错误：在路径 {input_path} 下没找到 .tfrecord 文件")
            return

        for f in files:
            try:
                df = self.process_file(f)
                if not df.empty:
                    save_name = os.path.join(self.output_dir, os.path.basename(f).replace('.tfrecord', '.csv'))
                    df.to_csv(save_name, index=False)
                    print(f"✅ 保存成功: {save_name} (数据行数: {len(df)})")
            except Exception as e:
                print(f"❌ 处理文件 {f} 时出错: {e}")

if __name__ == "__main__":
    # 确保文件名和你的实际文件一致
    INPUT_PATH = "data.tfrecord"  
    
    extractor = WaymoExtractor()
    extractor.run(INPUT_PATH)