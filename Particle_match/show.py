import csv
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 初始化图形和三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

flag = False
# 打开并读取 CSV 文件
with open('Track_3d_Data.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    x_vals_all, y_vals_all, z_vals_all = [], [], []
    for num, row in enumerate(reader):
        if row:  # 确保行不为空
            if num % 4 == 0:
                x_vals = [float(value) for value in row]
                x_vals_all += x_vals
            elif num % 4 == 1:
                y_vals = [float(value) for value in row]
                y_vals_all += y_vals
            elif num % 4 == 2:
                z_vals = [float(value) for value in row]
                z_vals_all += z_vals
        else:
            # 同一组数据内连线
            point_all=[]
            point_all = np.array([[x, y, z] for x, y, z in zip(x_vals_all, y_vals_all, z_vals_all)])
            ax.plot(point_all[:, 0], point_all[:, 1], point_all[:, 2])  # 使用 plot 连线
            if not flag:
                flag = True
                x_min, y_min, z_min = point_all.min(axis=0)
                x_max, y_max, z_max = point_all.max(axis=0)
            else:
                x_min = min(x_min, point_all[:, 0].min())
                y_min = min(y_min, point_all[:, 1].min())
                z_min = min(z_min, point_all[:, 2].min())
                x_max = max(x_max, point_all[:, 0].max())
                y_max = max(y_max, point_all[:, 1].max())
                z_max = max(z_max, point_all[:, 2].max())
            # 清空当前组数据
            x_vals_all, y_vals_all, z_vals_all = [], [], []

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([z_min, z_max])
ax.set_title('3D Trajectories')
ax.view_init(elev=-90, azim=-90)  # 设置视角
# 显示图形
plt.show() 