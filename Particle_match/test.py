import numpy as np
import cv2
import random
import os
from Particle_Set import Particle_Set
from match import Read_json
from tools import calcRcamEpipolars
canvas_match_left=np.zeros((1920,2592,3), dtype=np.uint8)
canvas_match_right=np.zeros((1920,2592,3), dtype=np.uint8)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Target Directory: ",os.getcwd())
json_path_left='../Particle_Info_Left_Data/'
json_path_right='../Particle_Info_Right_Data/'
img_path_left='C:/Users/Claudius/WorkingSpace/Article1/Particle_Analysis/LeftSight1/'
img_path_right='C:/Users/Claudius/WorkingSpace/Article1/Particle_Analysis/RightSight1/'
info_left=[]
info_right=[]
'''
for set in range(0,100):
    info_left.append(Particle_Set(f'INFO_{set}_LEFT',Read_json(json_path_left,set)))
    info_right.append(Particle_Set(f'INFO_{set}_RIGHT',Read_json(json_path_right,set)))
'''
'''
for i in range(100):
    for info in info_left[i].Particle_List:
        canvas_match_left=cv2.imread(info.img_path)
        cv2.circle(canvas_match_left, (int(info.x), int(info.y)), int(info.radius), (0, 255, 0), -1)
'''


def draw_epiline(canvas, k, b, color=(0, 0, 255)):
    h, w = canvas.shape[:2]
    x0, x1 = 0, w-1
    y0 = int(k * x0 + b)
    y1 = int(k * x1 + b)
    cv2.line(canvas, (x0, y0), (x1, y1), color, 2)

# 鼠标点击回调函数
def on_mouse(event, x, y, flags, param):
    global img_left_display, img_right_display, img_right_original
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

        # 在左图上画点击的点
        img_left_display = img_left_original.copy()
        img_right_display = img_right_original.copy()
        cv2.circle(img_left_display, (x, y), 8, (0, 255, 0), -1)

        # 计算对应右图的极线
        k, b,_ = calcRcamEpipolars(point, K1, K2, R, T)

        # 在右图上画极线
        draw_epiline(img_right_display, k, b)

        # 实时刷新显示
        cv2.imshow('Left Image', img_left_display)
        cv2.imshow('Right Image with Epiline', img_right_display)

def main():
    global img_left_original, img_right_original, img_left_display, img_right_display
    global K1, K2, R, T

    # 相机内参（示例，需要替换成自己的）
    K1 = np.array([[1000, 0, 1280],
                   [0, 1000, 960],
                   [0, 0, 1]])
    K2 = np.array([[1000, 0, 1280],
                   [0, 1000, 960],
                   [0, 0, 1]])

    # 相机外参（示例，需要替换成自己的）
    R = np.eye(3)
    T = np.array([1, 0, 0])  # 相机之间沿X轴1米

    # 读取左右图片
    img_left_path = '../LeftSight1/example_left.png'
    img_right_path = '../RightSight1/example_right.png'
    img_folder_left = '../LeftSight1/'
    img_folder_right = '../RightSight1/'
    files=[f for f in os.listdir(img_folder_left) if f.endswith('.bmp') ]
    img_left_original = cv2.imread(os.path.join(img_folder_left, files[100]))  # 替换为实际路径
    img_right_original = cv2.imread(os.path.join(img_folder_right, files[100]))

    if img_left_original is None or img_right_original is None:
        print("图像读取失败，请检查路径！")
        return

    img_left_display = img_left_original.copy()
    img_right_display = img_right_original.copy()

    # 创建窗口
    cv2.namedWindow('Left Image')
    cv2.namedWindow('Right Image with Epiline')

    # 绑定鼠标点击事件
    cv2.setMouseCallback('Left Image', on_mouse)

    # 显示初始图
    cv2.imshow('Left Image', img_left_display)
    cv2.imshow('Right Image with Epiline', img_right_display)

    print("请在左图上点击任意点，右图将实时画出对应极线。")
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # 按ESC键退出
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
