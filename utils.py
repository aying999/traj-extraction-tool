import yaml
import numpy as np
import pandas as pd

def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_box_coords(row, config):
    """
    计算旋转后的矩形框坐标
    参数:
        row: DataFrame 的一行数据
        config: 配置字典，用于获取默认尺寸
    """
    defaults = config['defaults']
    

    L = row.get('length', 0)
    W = row.get('width', 0)
    obj_type = str(row.get('type', '')).upper()
    

    if pd.isna(L) or L < 0.1:
        if 'PEDESTRIAN' in obj_type: 
            L, W = defaults['ped_length'], defaults['ped_width']
        elif 'CYCLIST' in obj_type: 
            L, W = defaults['cyc_length'], defaults['cyc_width']
        else: 
            L, W = defaults['car_length'], defaults['car_width']
        
    cx, cy, theta = row['x'], row['y'], row['heading']

    l2, w2 = L / 2.0, W / 2.0
    corners_rel = np.array([[-l2, -w2], [-l2, w2], [l2, w2], [l2, -w2]])
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    corners_rotated = corners_rel.dot(rot_matrix.T)
    corners_final = corners_rotated + np.array([cx, cy])


    x_coords = np.append(corners_final[:, 0], corners_final[0, 0])
    y_coords = np.append(corners_final[:, 1], corners_final[0, 1])
    
    return x_coords, y_coords