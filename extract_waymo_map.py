import os
import glob
import pandas as pd
import numpy as np
# 保持和之前一样的轻量级读取方式
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
        
        # 使用 tfrecord_iterator 读取，不依赖 tf.data
        loader = tfrecord_iterator(tfrecord_path)
        
        count = 0
        for record in loader:
            count += 1
            try:
                # 1. 反序列化 (解析二进制数据)
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record)
            except Exception as e:
                print(f"⚠️ 解析第 {count} 帧时出错: {e}")
                continue
            
            scenario_id = scenario.scenario_id
            
            # 2. 遍历该场景下的地图特征 (Map Features)
            # 注意：这里我们不再遍历 tracks，而是遍历 map_features
            for feature in scenario.map_features:
                feature_id = feature.id
                feature_type = feature.WhichOneof('feature_data')
                
                points = []
                map_type_str = 'UNKNOWN'
                
                # --- 提取不同类型的地图几何 ---
                
                # A. 车道 (Lane) - 包含中心线
                if feature_type == 'lane':
                    # 提取中心线 polyline
                    for p in feature.lane.polyline:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'LANE_CENTER' # 为了不混淆，我们标记为中心线
                    
                # B. 道路边缘 (Road Edge) - 马路牙子
                elif feature_type == 'road_edge':
                    for p in feature.road_edge.polyline:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'ROAD_EDGE'
                    
                # C. 道路标线 (Road Line) - 虚线/实线
                elif feature_type == 'road_line':
                    for p in feature.road_line.polyline:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'ROAD_LINE'
                
                # D. 停车标志 (Stop Sign) - 单点
                elif feature_type == 'stop_sign':
                    p = feature.stop_sign.position
                    points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'STOP_SIGN'
                
                # E. 人行道 (Crosswalk) - 多边形
                elif feature_type == 'crosswalk':
                    for p in feature.crosswalk.polygon:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'CROSSWALK'
                
                # F. 减速带 (Speed Bump) - 多边形
                elif feature_type == 'speed_bump':
                    for p in feature.speed_bump.polygon:
                        points.append({'x': p.x, 'y': p.y, 'z': p.z})
                    map_type_str = 'SPEED_BUMP'
                
                else:
                    continue # 跳过不关心的类型
                
                # 3. 将提取的点存入列表
                for i, pt in enumerate(points):
                    all_map_features.append({
                        'scenario_id': scenario_id,
                        'feature_id': feature_id,
                        'type': map_type_str,
                        'x': pt['x'],
                        'y': pt['y'],
                        'z': pt['z'],
                        'order': i  # 这一点非常重要，画线需要按顺序连起来
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
                    # 保存为 map_waymo.csv (或者 map_xxx.csv)
                    # 这里为了配合 app.py，我们强制保存为 map_waymo.csv，或者你可以保留原名
                    save_name = os.path.join(self.output_dir, "map_waymo.csv") 
                    
                    # 如果处理多个文件，这里可能需要 append 模式，或者只处理单文件
                    # 这里假设你只处理一个文件，直接覆盖
                    df.to_csv(save_name, index=False)
                    print(f"✅ 地图保存成功: {save_name} (数据点数: {len(df)})")
                else:
                    print("⚠️ 该文件未提取到地图数据")
            except Exception as e:
                print(f"❌ 处理文件 {f} 时出错: {e}")

if __name__ == "__main__":
    # 确保这里的文件名也是 data.tfrecord (或者你实际的文件名)
    INPUT_PATH = "data.tfrecord"  
    
    extractor = WaymoMapExtractor()
    extractor.run(INPUT_PATH)