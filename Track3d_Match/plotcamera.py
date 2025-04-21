import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_camera(ax, R, T, color='blue', name='Camera'):
    # 绘制相机坐标轴
    size = 100  # 坐标轴长度
    origin = T.reshape(3)
    x_axis = R @ np.array([size, 0, 0])
    y_axis = R @ np.array([0, size, 0])
    z_axis = R @ np.array([0, 0, size])
    
    ax.quiver(*origin, *x_axis, color='r', label='x')
    ax.quiver(*origin, *y_axis, color='g', label='y')
    ax.quiver(*origin, *z_axis, color='b', label='z')

    # 绘制图像平面（大致位置）
    img_plane_corners = np.array([
        [-size, -size, size],
        [ size, -size, size],
        [ size,  size, size],
        [-size,  size, size],
        [-size, -size, size]
    ]).T  # 3x5

    img_plane_corners = (R @ img_plane_corners) + T.reshape(3,1)
    ax.plot(img_plane_corners[0], img_plane_corners[1], img_plane_corners[2], color=color)
    ax.text(*origin, name, color=color)

def visualize_two_cameras(R, T):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 相机1（世界坐标原点）
    plot_camera(ax, np.eye(3), np.zeros((3,)), color='blue', name='Camera 1')

    # 相机2
    plot_camera(ax, R, T, color='orange', name='Camera 2')

    # 连线
    ax.plot([0, T[0]], [0, T[1]], [0, T[2]], 'k--')
    all_points = np.array([
        [0, 0, 0],
        T.reshape(3)
    ])
    max_range = np.max(np.ptp(all_points, axis=0))  # 取xyz三个方向中跨度最大的
    mid_point = np.mean(all_points, axis=0)

    # 设置x,y,z范围
    ax.set_xlim(mid_point[0] - max_range, mid_point[0] + max_range)
    ax.set_ylim(mid_point[1] - max_range, mid_point[1] + max_range)
    ax.set_zlim(mid_point[2] - max_range, mid_point[2] + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Two Cameras Visualization')
    ax.view_init(elev=20., azim=60)
    plt.show()
# 假设你有R, T
R = np.array([[0.747832245426745,	-0.00137646971907046,	0.663886314086309],
                [-0.0499508241855523	,0.997046660145766	,0.0583341465640672], 
                [-0.662005927362570	,-0.0767858243641788,	0.745554886855111]], dtype=np.float64)
T = np.array([-332.797331625628,5.51661638734196,197.141393142023]).T
visualize_two_cameras(R, T)
