import cv2, json
import mediapipe as mp
import numpy as np
from moviepy.editor import *

path = 'demo4'
video = VideoFileClip(path + '.mp4')
audio = video.audio
audio.write_audiofile(path + '.mp3')


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
dic={'LEFTELBOW':[],
'RIGHTELBOW':[],
'LEFTSHOULDER':[],
'RIGHTSHOULDER':[],
'LEFTHIP':[],
'RIGHTHIP':[],
'LEFTKNEE':[],
'RIGHTKNEE':[]
}
DIC={'LEFTELBOW':[],
'RIGHTELBOW':[],
'LEFTSHOULDER':[],
'RIGHTSHOULDER':[],
'LEFTHIP':[],
'RIGHTHIP':[],
'LEFTKNEE':[],
'RIGHTKNEE':[]
}

# 分數的list
list=[0]
i = 0


# 算兩邊的夾角和
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 



def detect(img):

    # 將圖檔資料BGR轉成RGB格式
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 預測此圖姿勢
    results = pose.process(image)
    image.flags.writeable = True

    # 轉回BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
    # 取出節點
    # try:
    landmarks = results.pose_landmarks.landmark
    
    # 找座標，形式為[x座標, y座標]
    LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    LEFT_KNEE=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    LEFT_ANKLE=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Calculate angle
    # 肩-手肘-手腕
    LEFTELBOW_angle = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    RIGHTELBOW_angle = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    # 手肘-肩-臀
    LEFTSHOULDER_angle = calculate_angle(LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
    RIGHTSHOULDER_angle = calculate_angle(RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)
    # 肩-臀-膝
    LEFTHIP_angle = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    RIGHTHIP_angle = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    # 臀-膝-腳踝
    LEFTKNEE_angle = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    RIGHTKNEE_angle = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    
    # 將角度存進列表裡
    dic['LEFTELBOW'].append(LEFTELBOW_angle)
    dic['RIGHTELBOW'].append(RIGHTELBOW_angle)
    dic['LEFTSHOULDER'].append(LEFTSHOULDER_angle)
    dic['RIGHTSHOULDER'].append(RIGHTSHOULDER_angle)
    dic['LEFTHIP'].append(LEFTHIP_angle)
    dic['RIGHTHIP'].append(RIGHTHIP_angle)
    dic['LEFTKNEE'].append(LEFTKNEE_angle)
    dic['RIGHTKNEE'].append(RIGHTKNEE_angle)




cap = cv2.VideoCapture(path + '.mp4')
i = 0

size = (int(cap.get(3)), int(cap.get(4)))
fps = int(cap.get(5))
save = cv2.VideoWriter('detect_' + path + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, img = cap.read()
        i += 1
        # print('now', i)
        if ret == True:
            try:
                detect(img)

                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img)

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                save.write(img)
            except:
                pass
      
        else:
            break
cap.release()
save.release()

import json
with open(path + '.json', 'w') as f:
    json.dump(dic, f, indent = 2)

