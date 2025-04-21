import numpy as np
import os
import json
import cv2
import screeninfo
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Data import Data
class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __str__(self):
        return "({0},{1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)
    def __mul__(self, other):
        x = self.x * other.x
        y = self.y * other.y
        return Point(x, y)
    def __len__(self):
        return math.sqrt(self.x**2+self.y**2)
    def __truediv__(self, other):
        if other.x == 0 or other.y == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")   
        x = self.x / other.x
        y = self.y / other.y
    def dot(self, other):
        return self.x * other.x + self.y * other.y
def Read_json(json_path,img_serial_num):
    
    sort_img_lst=sorted([f for f in os.listdir(json_path) if f.endswith('json')])
    beginname=sort_img_lst[0]
    begin_serial=int(beginname.split('.')[0].split('_')[1])
    img_name=sort_img_lst[img_serial_num-begin_serial]
    jsonname=os.path.join(json_path,img_name)
    unix_path = jsonname.replace("\\", "/")  # 替换 \ 为 /
    Info = []
    with open(unix_path, 'r', encoding='utf-8') as f:
        Info= json.load(f)
    return Info
class Particle_Set(object):
    def __init__(self,name,json_set):
        self.name=name
        self.Time_Stamp=json_set[0]['img_num']
        self.Image_Path=json_set[0]['img_path']
        self.Particle_Num=len(json_set)
        self.Particle_List=[]
        for iter,entry in enumerate(json_set):
            coordinate=Point(entry['x']+entry['x_in'],entry['y']+entry['y_in'])
            radius=entry['r']
            iternum=iter
            self.Particle_List.append([coordinate,radius,iternum])
    def plot(self,canvas,color=(0,255,0)):
        for iter in self.Particle_List:
            coordinate=iter[0]
            radius=iter[1]
            iternum=iter[2]
            cv2.circle(canvas,(int(coordinate.x),int(coordinate.y)),int(radius),color,2)
            #cv2.putText(canvas,str(iternum),(int(coordinate.x),int(coordinate.y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
json_path_left='C:/Users/Claudius/WorkingSpace/Article1/Particle_Info_Left_Data'
json_path_right='C:/Users/Claudius/WorkingSpace/Article1/Particle_Info_Right_Data'
info_left=[]
info_right=[]
for set in range(2000,2100):
    info_left.append(Particle_Set(f'INFO_{set}_LEFT',Read_json(json_path_left,set)))
    info_right.append(Particle_Set(f'INFO_{set}_RIGHT',Read_json(json_path_right,set)))

r_lst= [info_left[99].Particle_List[i][1] for i in range(info_left[99].Particle_Num)]
r_lst = np.array(r_lst)*25
plt.figure(figsize=(10, 6))
sns.histplot(data=r_lst,kde=True, bins=30, color='blue', alpha=0.6)
plt.title('Histogram of Particle Radii')
plt.xlabel('Radius/um')
plt.ylabel('Count/times')
plt.savefig('./Hist_radius.png', dpi=300, bbox_inches='tight')
plt.show()
