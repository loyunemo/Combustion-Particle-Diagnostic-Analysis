import numpy as np
from Point import Point
import cv2
def point2dTo3d(point_2d, z, K):
    return np.array([(point_2d[0]-K[0,2])*z/K[0,0], (point_2d[1]-K[1,2])*z/K[1,1], z], dtype=np.float64).T
def calcRcamEpipolar(point_2d, K1, K2, R, T):
    point_2dt = np.array([point_2d.x,point_2d.y], dtype=np.float64).T
    point1_3d = R@np.array([0,0,0]).T + T
    point2_3d = R@np.array(point2dTo3d(point_2dt, 1, K1)).T + T
    # print(point1_3d, point2_3d)
    point1_2d = K2@point1_3d / point1_3d[2]
    point2_2d = K2@point2_3d / point2_3d[2]
    print(point1_2d, point2_2d)
    # print(point1_2d, point2_2d)
    k = (point1_2d[1] - point2_2d[1]) / (point1_2d[0] - point2_2d[0])
    b = (point1_2d[0]*point2_2d[1] - point2_2d[0]*point1_2d[1]) / (point1_2d[0] - point2_2d[0])
    # print(k,b)
    return k, b, point2_3d, point1_3d
def calcRcamEpipolars(point_2dLst, K1, K2, R, T):
    linelst=[]
    point2_3dLst=[]
    point1_3dLst=[]
    for point_2d in point_2dLst.tracklst:
        k, b, point2_3d, point1_3d=calcRcamEpipolar(point_2d.coordinate, K1, K2, R, T)
        linelst.append((k,b))
        point2_3dLst.append(point2_3d)
        point1_3dLst.append(point1_3d)
    #print("[LINE]:",linelst)
    return linelst
def vertex_ditance(TrackObject,Line1,Line2,Line3):
    particle0=TrackObject.tracklst[0]
    particle1=TrackObject.tracklst[1]
    particle2=TrackObject.tracklst[2]
    distance0=abs(Line1[0]*particle0.coordinate.x+Line1[1]-particle0.coordinate.y)/np.sqrt(Line1[0]**2+1)
    distance1=abs(Line2[0]*particle1.coordinate.x+Line2[1]-particle1.coordinate.y)/np.sqrt(Line2[0]**2+1)
    distance2=abs(Line3[0]*particle2.coordinate.x+Line3[1]-particle2.coordinate.y)/np.sqrt(Line3[0]**2+1)
    return distance0, distance1, distance2
def triangulate_point(p1, p2, K1, K2, R, T):
    # 将图像坐标归一化
    B=np.zeros(3)
    ds=7.8/1000
    B[2]=K1[1,1]*ds*(K2[1,1]*ds*T[1]-p2[1]*T[2])/(p2[1]*(R[2,0]*p1[0]*K1[1,1]/K1[0,0]+R[2,1]*p1[1]+R[2,2]*K1[1,1]*ds)-K2[1,1]*ds*(R[1,0]*p1[0]*K1[1,1]/K1[0,0]+R[1,1]*p1[1]+K1[1,1]*ds*R[1,2]))
    B[1]=B[2]*p1[0]/K1[1,1]/ds
    B[0]=B[2]*p1[1]/K1[0,0]/ds  
    return B
def Reconstruct_3d_point(TrackObjectLeft,TrackObjectRight,K1,K2,R,T):
    Lst=[]
    for PointL,PointR in zip(TrackObjectLeft.tracklst,TrackObjectRight.tracklst):
        point_2dL=np.array([PointL.coordinate.x,PointL.coordinate.y]).T
        point_2dR=np.array([PointR.coordinate.x,PointR.coordinate.y]).T
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, T.reshape(3, 1)))
        
        point_3D=cv2.triangulatePoints(P1,P2,point_2dL, point_2dR)
        point_3d = point_3D[:3] / point_3D[3]  # 归一化
        point_3d=[point_3d[0][0],point_3d[1][0],point_3d[2][0]]
        #print(point_3d.T.shape)
        Lst.append(point_3d)
    return Lst
def Reconstruct_3d_point_Single(PointL,PointR,K1,K2,R,T):

    point_2dL=np.array([PointL.coordinate.x,PointL.coordinate.y]).T
    point_2dR=np.array([PointR.coordinate.x,PointR.coordinate.y]).T
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))
    
    point_3D=cv2.triangulatePoints(P1,P2,point_2dL, point_2dR)
    point_3d = point_3D[:3] / point_3D[3]  # 归一化
    point_3d=[point_3d[0][0],point_3d[1][0],point_3d[2][0]]

    return point_3d
def project_to_pixel(B, K1):
    X, Y, Z = B
    fx = K1[0, 0]
    fy = K1[1, 1]
    cx = K1[0, 2]
    cy = K1[1, 2]
    
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return Point(u, v)
def project_to_camera2(B, K2, R, T):
    # B: (3,) 世界坐标点
    # R: (3,3) 旋转矩阵（世界到相机2）
    # T: (3,)  平移向量（世界到相机2）
    
    # Step 1: 世界坐标到相机2坐标
    B_cam2 = R @ B + T  # 注意是R乘B加T
    
    X_c, Y_c, Z_c = B_cam2
    
    # Step 2: 投影到像素
    fx = K2[0, 0]
    fy = K2[1, 1]
    cx = K2[0, 2]
    cy = K2[1, 2]
    
    u = fx * X_c / Z_c + cx
    v = fy * Y_c / Z_c + cy
    return Point(u, v)
def ReProjection_To_Left(ReconstructPoint1,ReconstructPoint2,ReconstructPoint3,K1,R,T):
    p1,p2,p3=project_to_pixel(ReconstructPoint1,K1),project_to_pixel(ReconstructPoint2,K1),project_to_pixel(ReconstructPoint3,K1)
    return p1,p2,p3

def ReProjection_To_Right(ReconstructPoint1,ReconstructPoint2,ReconstructPoint3,K2,R,T):
    p1,p2,p3=project_to_camera2(ReconstructPoint1,K2,R, T),project_to_camera2(ReconstructPoint2,K2,R, T),project_to_camera2(ReconstructPoint3,K2,R, T)
    return p1,p2,p3