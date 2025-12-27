import pandas as pd
import matplotlib.pyplot as plt
import random

# 1. 读取数据 (只读前 10万行防止内存爆炸，或者读取全部)
print("⏳ 正在读取数据...")
df = pd.read_csv("output/data.csv")

# 2. 随机选择一个场景 ID
unique_scenarios = df['scenario_id'].unique()
chosen_id = random.choice(unique_scenarios)
print(f"🎨 正在绘制场景: {chosen_id}")

# 3. 过滤出该场景的数据
scene_df = df[df['scenario_id'] == chosen_id]

# 4. 绘图
plt.figure(figsize=(10, 10))
plt.title(f"Waymo Motion Scenario: {chosen_id}")
plt.xlabel("Global X (m)")
plt.ylabel("Global Y (m)")
plt.axis('equal') # 保证比例尺一致，否则路是歪的

# 画其他车辆 (灰色)
others = scene_df[~scene_df['is_ego']]
# 按 track_id 分组画，保证轨迹是连贯的线
for track_id, track_data in others.groupby('track_id'):
    plt.plot(track_data['x'], track_data['y'], color='gray', alpha=0.5, linewidth=1)

# 画自车 (红色，加粗)
ego = scene_df[scene_df['is_ego']]
plt.plot(ego['x'], ego['y'], color='red', linewidth=3, label='Ego Vehicle')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# 5. 保存图片
output_file = f"vis_{chosen_id}.png"
plt.savefig(output_file, dpi=150)
print(f"✅ 图片已保存为: {output_file}")