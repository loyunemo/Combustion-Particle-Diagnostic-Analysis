from Particle_Set import Particle_Set
from match import Read_json
from match import Match_Evaluate
from match import Similarity_Predict_Evaluate
from Match_Planner import Match_Planner,Track_3d_Match
import cv2
import numpy as np
import random
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Target Directory: ",os.getcwd())
json_path_left='../Particle_Info_Left_Data/'
json_path_right='../Particle_Info_Right_Data/'
img_path_left='../LeftSight1/'
img_path_right='../RightSight1/'
info_left=[]
info_right=[]
for set in range(0,100):
    info_left.append(Particle_Set(f'INFO_{set}_LEFT',Read_json(json_path_left,set)))
    info_right.append(Particle_Set(f'INFO_{set}_RIGHT',Read_json(json_path_right,set)))
canvas_left=np.zeros((1920,2592,3), dtype=np.uint8)
canvas_match_left=np.zeros((1920,2592,3), dtype=np.uint8)
#canvas_right=np.zeros((2592, 1920, 3), dtype=np.uint8)
colorset1 = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
#colorset=[(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,128),(128,0,128),(128,128,0),(0,128,128)]
Match_Planner1=Match_Planner(img_path_left)
Match_Planner2=Match_Planner(img_path_right)

Match_Planner1.Base_Track(info_left[0],info_left[1],info_left[2])
Match_Planner2.Base_Track(info_right[0],info_right[1],info_right[2])
#for i in range(96):
    #result=Match_Planner1.Track_Sort(info_left[i],info_left[i+1],info_left[i+2],enable_track=False)
#    Match_Planner1.Track_Sort_Merge(info_left[i+1],info_left[i+2],info_left[i+3])
'''
for iter in result:
    cv2.line(canvas_match_left,(int(info_left[i].Particle_List[iter[0]].coordinate.x),int(info_left[i].Particle_List[iter[0]].coordinate.y)),(int(info_left[i+1].Particle_List[iter[1]].coordinate.x),int(info_left[i+1].Particle_List[iter[1]].coordinate.y)),(0,0,255),1)
    cv2.line(canvas_match_left,(int(info_left[i+1].Particle_List[iter[1]].coordinate.x),int(info_left[i+1].Particle_List[iter[1]].coordinate.y)),(int(info_left[i+2].Particle_List[iter[2]].coordinate.x),int(info_left[i+2].Particle_List[iter[2]].coordinate.y)),(255,245,0),1)
    Match_Evaluate(info_left[i],info_left[i+1],canvas_match_left,None,drawflagbefore=True,drawflagafter=True,showflag=False,colorbefore=colorset1[i],colorafter=colorset1[i+1])
    Match_Evaluate(info_left[i+1],info_left[i+2],canvas_match_left,None,drawflagbefore=False,drawflagafter=True,showflag=True,color=colorset1[i+1],colorafter=colorset1[i+2])
'''
#Match_Planner1.write_tracks_to_csv('trackdata.csv')
#for i in range(97):
    #Match_Planner1.Track_After(info_left[i+3])
    #Match_Planner2.Track_After(info_right[i+3])
for i in range(48):
    Match_Planner1.Track_After_2frame(info_left[i*2+3],info_left[i*2+4])
'''    
Match_Planner1.Info()
for i in Match_Planner1.OutTrack:
    i.Plot_Track(canvas_left)
for i in Match_Planner1.TrackLst:
    i.Plot_Track(canvas_left)
cv2.imwrite('Track.jpg',canvas_left)
'''
Track_3d_Match(Match_Planner1,Match_Planner2,info_left,info_right)
'''    #info_right[0].Particle_List[i].plot(canvas_right,colorset1[i])
result=Match_Planner1.Track_Sort(info_left[0],info_left[1],info_left[2])
for i in result:
    cv2.line(canvas_match_left,(int(info_left[0].Particle_List[i[0]].coordinate.x),int(info_left[0].Particle_List[i[0]].coordinate.y)),(int(info_left[1].Particle_List[i[1]].coordinate.x),int(info_left[1].Particle_List[i[1]].coordinate.y)),(0,0,255),1)
    cv2.line(canvas_match_left,(int(info_left[1].Particle_List[i[1]].coordinate.x),int(info_left[1].Particle_List[i[1]].coordinate.y)),(int(info_left[2].Particle_List[i[2]].coordinate.x),int(info_left[2].Particle_List[i[2]].coordinate.y)),(255,245,0),1)
Match_Evaluate(info_left[0],info_left[1],canvas_match_left,90,drawflagbefore=True,drawflagafter=True,showflag=False,colorbefore=colorset1[10],colorafter=colorset1[11])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,84,drawflagbefore=False,drawflagafter=True,showflag=True,color=colorset1[0],colorafter=colorset1[12])
'''
#print(result)
#cv2.imwrite('Debug.jpg',canvas_left)
#show_resized_image_auto(canvas_left)
#result1=Similarity_Predict_Evaluate(info_left[0],info_left[1],info_left[2],canvas_match_left,drawflag1=False,drawflag2=False)
#result2=Similarity_Predict_Evaluate(info_left[2],info_left[1],info_left[0],canvas_match_left,drawflag1=False,drawflag2=False)
#result2_reversed = [list(reversed(i)) for i in result2]
#result2_reversed.sort(key=lambda x: x[0])
#result_selected = []

#for i in result2_reversed:
#    if i in result1:
#        result_selected.append(i)

'''
for i in result1:
    cv2.line(canvas_match_left,(int(info_left[0].Particle_List[i[0]][0].x),int(info_left[0].Particle_List[i[0]][0].y)),(int(info_left[1].Particle_List[i[1]][0].x),int(info_left[1].Particle_List[i[1]][0].y)),(0,0,255),1)
    cv2.line(canvas_match_left,(int(info_left[1].Particle_List[i[1]][0].x),int(info_left[1].Particle_List[i[1]][0].y)),(int(info_left[2].Particle_List[i[2]][0].x),int(info_left[2].Particle_List[i[2]][0].y)),(255,128,255),1)
'''
#cv2.line(canvas_match_left,(int(info_left[0].Particle_List[result1[0][0]][0].x),int(info_left[0].Particle_List[result1[0][0]][0].y)),(int(info_left[1].Particle_List[result1[0][1]][0].x),int(info_left[1].Particle_List[result1[0][1]][0].y)),(255,128,255),1)
#Similarity_Predict_Evaluate(info_left[1],info_left[2],info_left[3],canvas_match_left,drawflag1=False,drawflag2=True)
Match_Evaluate(info_left[0],info_left[1],canvas_match_left,90,drawflagbefore=True,drawflagafter=True,showflag=False,colorbefore=colorset1[10],colorafter=colorset1[11])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,84,drawflagbefore=False,drawflagafter=True,showflag=False,color=colorset1[0],colorafter=colorset1[12])
#Match_Evaluate(info_left[2],info_left[3],canvas_match_left,85,drawflagbefore=False,drawflagafter=True,showflag=False,color=colorset1[8],colorafter=colorset1[7])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,81,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,88,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,71,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,76,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
cv2.imwrite('Match.jpg',canvas_match_left)
print("done")