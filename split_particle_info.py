import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 读取 Particle_info_right.json 文件
input_file1 = './Particle_Detect/Particle_info_left.json'
output_dir1 = 'Particle_Info_Left_Data'
input_file2 = './Particle_Detect/Particle_info_right.json'
output_dir2 = 'Particle_Info_Right_Data'
# 创建存储分离文件的目录

os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

with open(input_file1, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

# 根据 img_num 字段分组并存储
grouped_data = {}
for entry in data:
    img_num = entry['img_num']
    if img_num not in grouped_data:
        grouped_data[img_num] = []
    grouped_data[img_num].append(entry)

# 将每个 img_num 的数据存储为单独的 JSON 文件
for img_num, entries in grouped_data.items():
    output_file = os.path.join(output_dir1, f'img_{img_num}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)

print(f"Data has been split and stored in '{output_dir1}'")

with open(input_file2, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

# 根据 img_num 字段分组并存储
grouped_data = {}
for entry in data:
    img_num = entry['img_num']
    if img_num not in grouped_data:
        grouped_data[img_num] = []
    grouped_data[img_num].append(entry)

# 将每个 img_num 的数据存储为单独的 JSON 文件
for img_num, entries in grouped_data.items():
    output_file = os.path.join(output_dir2, f'img_{img_num}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)

print(f"Data has been split and stored in '{output_dir2}'")
