import numpy as np
from Data import Data
from PIL import Image
from read import Read_Image
from Seg import Kmeans_Seg,MOG2
import os
import json
import cv2

Left_Image_Path='C:/Users/Claudius/WorkingSpace/Article1/0722experiment2/51/BMP192.168.8.51-20240707-072633'
Kmeans=Kmeans_Seg()
Back_SubSeg=MOG2(Left_Image_Path)
Back_SubSeg.pretrain(3000,3999)
t=Back_SubSeg.MOG2_Seg(4000,False)
Image_Info=[]
Radius_Info=[]
for iter in t:
    sth=iter.Data_Write()
    path=os.path.join(Left_Image_Path,Back_SubSeg.Imagelist[sth[-1]])
    Single_imginfo,Single_radiusinfo=Kmeans.Seg(path,sth[0],sth[1],sth[2],sth[3],False)
    Image_Info.append(Single_imginfo)
    Radius_Info.append(Single_radiusinfo)
with open('Radius_Info.json', 'w', encoding='utf-8') as f:
    for info in Radius_Info:
        for t in info:    
            if t[2]<1e-4:
                continue
            disk = {'x': t[0] + t[3], 'y': t[1] + t[4], 'r': t[2]}
            disk ={'x':t[0],'y':t[1],'r':t[2],'x_in':t[3],'y_in':t[4]}
            json.dump(disk, f, ensure_ascii=False)
            f.write('\n')  # 添加换行符
print("done")

# 读取 Radius_Info.json 并在图片中绘制圆
with open('Radius_Info.json', 'r', encoding='utf-8') as f:
    radius_data = [json.loads(line) for line in f if line.strip()]

# 指定图片路径
image_path = os.path.join(Left_Image_Path, Back_SubSeg.Imagelist[4000])  # 替换为需要绘制的图片
image = cv2.cvtColor(Read_Image(image_path),cv2.COLOR_GRAY2RGB)

if image is None:
    print(f"Error: Unable to read image at {image_path}")
else:
    for disk in radius_data:
        center = (int(disk['x']+disk['x_in']), int(disk['y']+disk['y_in']))
        radius = int(disk['r'])
        out=cv2.circle(image, center, radius, (0, 255, 0), 2)  # 绿色圆圈

    # 显示绘制结果
    cv2.imwrite('output_image.jpg', out)  # 保存绘制结果
    cv2.imshow('Image with Circles', cv2.resize(out, (image.shape[1] // 2, image.shape[0] // 2)))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()