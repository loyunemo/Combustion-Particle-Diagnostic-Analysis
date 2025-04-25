import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D绘图

from tqdm import tqdm  # 用来显示进度条
import scipy.sparse as sp
def plot_voxel(volume, threshold=0.5, downsample=4):
    """
    volume: 3D体积数据，shape = (Nx, Ny, Nz)
    threshold: 阈值，体素值大于这个就绘制
    downsample: 降采样倍数，减小绘制负担
    """
    Nx, Ny, Nz = volume.shape
    
    # 降采样
    vol_small = volume[::downsample, ::downsample, ::downsample]

    # 归一化
    vol_norm = (vol_small - vol_small.min()) / (vol_small.max() - vol_small.min() + 1e-8)
    
    # 创建体素掩膜
    filled = vol_norm > threshold

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(filled, facecolors='cyan', edgecolor='k', linewidth=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Plot')
    
    plt.tight_layout()
    plt.show()
def load_and_resize_images(proj_x, proj_y, target_size=(432, 320)):
    
    
    proj_x_small = cv2.resize(proj_x, target_size, interpolation=cv2.INTER_AREA)
    proj_y_small = cv2.resize(proj_y, target_size, interpolation=cv2.INTER_AREA)
    
    return proj_x_small, proj_y_small
def create_sparse_system_matrix(volume_shape, projections):
    Nx, Ny, Nz = volume_shape
    data = []
    rows = []
    cols = []
    b = []

    row_idx = 0

    for direction, proj in projections:
        if direction == 'x':
            for j in range(Ny):
                for k in range(Nz):
                    for i in range(Nx):
                        idx = i * Ny * Nz + j * Nz + k
                        data.append(1.0)
                        rows.append(row_idx)
                        cols.append(idx)
                    b.append(proj[j, k])
                    row_idx += 1

        elif direction == 'y':
            for i in range(Nx):
                for k in range(Nz):
                    for j in range(Ny):
                        idx = i * Ny * Nz + j * Nz + k
                        data.append(1.0)
                        rows.append(row_idx)
                        cols.append(idx)
                    b.append(proj[i, k])
                    row_idx += 1

    A_sparse = sp.csr_matrix((data, (rows, cols)), shape=(row_idx, Nx*Ny*Nz))
    b = np.array(b)
    return A_sparse, b
def ART_vectorized(A_sparse, b, volume_shape, num_iter=10, relax=1.0):
    m, n = A_sparse.shape
    x = np.zeros(n)

    # 预先计算每行的范数平方（只计算一次！加速）
    row_norm_sq = np.array(A_sparse.power(2).sum(axis=1)).flatten()

    for it in range(num_iter):
        pbar = tqdm(total=m, desc=f'Iteration {it+1}/{num_iter}')
        
        # 全量预测
        Ax = A_sparse @ x
        
        # 残差
        residual = b - Ax
        
        # 更新量 delta_x = A.T * (residual / row_norm)
        scale = np.divide(residual, row_norm_sq + 1e-8)  # 防止除零
        delta_x = A_sparse.transpose().dot(scale)

        # 更新解
        x += relax * delta_x

        pbar.update(m)
        pbar.close()

    return x.reshape(volume_shape)

def undistort_image(img, K, D):
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    img_undistorted = cv2.undistort(img, K, D, None, new_K)
    return img_undistorted, new_K

def pixel_to_ray(u, v, K):
    inv_K = np.linalg.inv(K)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)  # (h, w, 3)
    rays = (inv_K @ uv1[..., None])[..., 0]            # (h, w, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)  # 归一化方向向量
    return rays

def main():
    # 1. 读入图像（假设输入图像是8位单通道）
    img1=cv2.imread('C:/Users/Claudius/WorkingSpace/Article1/Particle_Analysis/LeftSight1/2024-07-07 01-05-54.282302_2000.bmp', cv2.IMREAD_GRAYSCALE)
    
    img2 = cv2.imread('C:/Users/Claudius/WorkingSpace/Article1/Particle_Analysis/RightSight1/2024-07-07 00-02-29.893600_2000.bmp', cv2.IMREAD_GRAYSCALE)
    h, w = img2.shape[:2]
    print("Image shape:", img2.shape)
    # 2. 相机内参和畸变参数
    K1s = np.array([[1.151812867953681e+04, 0, 2.930281306554746e+02],
                    [0, 1.144065639977414e+04, 9.299614621342721e+02], 
                    [0, 0, 1]], dtype=np.float64)
    
    K2s = np.array([[1.416281370757014e+04, 0, 2.186893663098017e+03],
                    [0, 1.397124917519382e+04, -2.934480724268153e+02], 
                    [0, 0, 1]], dtype=np.float64)
    
    D1 = np.array([-1.03599173650834, 30.4343730533085, -0.0175971456828163, 0.0775278591952446], dtype=np.float64)
    D2 = np.array([1.49649840089790, -14.9214987001899, -0.0891119097877299, 0.0319277395780453], dtype=np.float64)
    img1=cv2.undistort(img1, K1s, D1)
    # 3. 相机外参
    R2 = np.array([[0.747832245426745, -0.00137646971907046, 0.663886314086309],
                   [-0.0499508241855523, 0.997046660145766, 0.0583341465640672], 
                   [-0.662005927362570, -0.0767858243641788, 0.745554886855111]], dtype=np.float64)
    T2 = np.array([[-332.797331625628], [5.51661638734196], [197.141393142023]])

    # 4. 第一个相机的位置：假设世界坐标系原点
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))

    # 5. 去畸变相机2的图像
    img2_undistorted, new_K2 = undistort_image(img2, K2s, D2)
    #cv2.imshow('Undistorted Image', img2_undistorted)
    #cv2.waitKey(0)
    # 6. 生成像素网格 (u,v)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays2 = pixel_to_ray(u, v, new_K2)  # (h, w, 3)

    # 7. 射线方向变换到世界坐标系
    R2_inv = R2.T
    ray_dirs_world = rays2 @ R2_inv.T  # (h, w, 3)
    cam2_center_world = (-R2_inv @ T2).reshape(-1)  # (3,)

    # 8. 定义目标平面（垂直于相机1成像平面）
    plane_normal = np.array([0, 0, 1], dtype=np.float64)
    plane_point = np.array([0, 0, 0], dtype=np.float64)

    # 9. 射线和平面求交点
    ndotu = ray_dirs_world @ plane_normal  # (h, w)
    
    # 避免除以0（射线平行于平面）
    ndotu = np.where(np.abs(ndotu) < 1e-6, 1e-6, ndotu)

    d = np.sum(plane_normal * plane_point)
    si = -(cam2_center_world @ plane_normal - d) / ndotu  # (h, w)
    
    intersection_pts = cam2_center_world[None, None, :] + ray_dirs_world * si[..., None]  # (h, w, 3)

    # 10. 提取平面上的 (x, y)
    x_new = intersection_pts[..., 0]
    y_new = intersection_pts[..., 1]

    # 11. 归一化到新图尺寸
    x_min, x_max = np.min(x_new), np.max(x_new)
    y_min, y_max = np.min(y_new), np.max(y_new)

    new_w, new_h = 2592, 1920  # 目标图像大小

    x_norm = (x_max - x_new) / (x_max - x_min) * (new_w - 1)
    y_norm = (y_max - y_new) / (y_max - y_min) * (new_h - 1)  # 翻转 y 坐标

    # 12. remap映射
    map_x = x_norm.astype(np.float32)
    map_y = y_norm.astype(np.float32)

    result = cv2.remap(img2_undistorted, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 13. 确保输出为 8 位单通道
    result = np.clip(result, 0, 255).astype(np.uint8)  # 确保图像值在0-255之间，并转换为8位无符号整数
    proj_x_small, proj_y_small = load_and_resize_images(img1, result, target_size=(512, 512))
    
    # 2. 创建稀疏系统矩阵
    Nx, Ny, Nz = 432, 512, 432
    A_sparse, b = create_sparse_system_matrix((Nx, Ny, Nz), [('x', proj_x_small), ('y', proj_y_small)])
    
    # 3. ART重建（向量化版）
    recon_volume = ART_vectorized(A_sparse, b, (Nx, Ny, Nz), num_iter=10, relax=1.0)
    plot_voxel(recon_volume, threshold=0.5, downsample=2)
if __name__ == "__main__":
    main()
