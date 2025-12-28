import os
import glob
import pandas as pd
import numpy as np
from tfrecord.reader import tfrecord_iterator
from waymo_open_dataset.protos import scenario_pb2

class WaymoMapExtractor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_file(self, tfrecord_path):
        print(f"🚀 正在处理地图数据: {os.path.basename(tfrecord_path)}")
        all_map_features = []
        
        
        loader = tfrecord_iterator(tfrecord_path)
        
        count = 0
        for record in loader:
            count += 1
            try:
            
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record)
            except Exception as e:
                print(f"⚠️ 解析第 {count} 帧时出错: {e}")
                continue
            
            scenario_id = scenario.scenario_id
     
            for feature in scenario.map_features:
                feature_id = feature.id
                feature_type = feature.WhichOneof('feature_data')
                
                points = []
                map_type_str = 'UNKNOWN'
                
         
                if feature_type == 'lane':
                    for p in feature.lane.polyline:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'LANE_CENTER' 
                    
              
                elif feature_type == 'road_edge':
                    for p in feature.road_edge.polyline:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'ROAD_EDGE'
                    
               
                elif feature_type == 'road_line':
                    for p in feature.road_line.polyline:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'ROAD_LINE'
                
              
                elif feature_type == 'stop_sign':
                    p = feature.stop_sign.position
                    points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'STOP_SIGN'
                
                
                elif feature_type == 'crosswalk':
                    for p in feature.crosswalk.polygon:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'CROSSWALK'
                
              
                elif feature_type == 'speed_bump':
                    for p in feature.speed_bump.polygon:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'SPEED_BUMP'
                
                else:
                    continue 
                
           
                for i, pt in enumerate(points):
                    all_map_features.append({
                        'scenario_id': scenario_id,
                        'feature_id': feature_id,
                        'type': map_type_str,
                        'x': pt['x'],
                        'y': pt['y'],
                        'z': pt['z'],
                        'order': i  
                    })

        print(f"   -> 解析完成，包含 {count} 个场景")
        return pd.DataFrame(all_map_features)

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
                    
                    save_name = os.path.join(self.output_dir, "map_waymo.csv") 
                    
                 
                    df.to_csv(save_name, index=False)
                    print(f"✅ 地图保存成功: {save_name} (数据点数: {len(df)})")
                else:
                    print("⚠️ 该文件未提取到地图数据")
            except Exception as e:
                print(f"❌ 处理文件 {f} 时出错: {e}")

if __name__ == "__main__":

    INPUT_PATH = "data.tfrecord"  
    
    extractor = WaymoMapExtractor()
    extractor.run(INPUT_PATH)