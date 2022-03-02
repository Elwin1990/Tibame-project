import cv2, json, pygame, time
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

mp_drawing_styles = mp.solutions.drawing_styles

path = 'demo4'

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


    # 算分數
def angle_diff(x, y):
    if x != None and y != None:
        return abs(x-y)
    else:
        return None

def mean(list):
    i = 0
    total = 0
    for x in list:
        if x != None:
            total += x
            i += 1
    return int(x/i)

def score(i):
    # 各部位角度差之總和
    Difference = abs(int(DIC['LEFTELBOW'][i])     - int(dic['LEFTELBOW'][i])) \
               + abs(int(DIC['RIGHTELBOW'][i])    - int(dic['RIGHTELBOW'][i])) \
               + abs(int(DIC['LEFTSHOULDER'][i])  - int(dic['LEFTSHOULDER'][i])) \
               + abs(int(DIC['RIGHTSHOULDER'][i]) - int(dic['RIGHTSHOULDER'][i])) \
               + abs(int(DIC['LEFTHIP'][i])       - int(dic['LEFTHIP'][i])) \
               + abs(int(DIC['RIGHTHIP'][i])      - int(dic['RIGHTHIP'][i])) \
               + abs(int(DIC['LEFTKNEE'][i])      - int(dic['LEFTKNEE'][i])) \
               + abs(int(DIC['RIGHTKNEE'][i])     - int(dic['RIGHTKNEE'][i])) 
    
    # 
    if Difference <= 70:
        list.append(int(list[i])+5)
        return list[i]
                                     
    elif Difference <= 140:
        list.append(int(list[i])+3)
        return list[i]
          
    elif Difference <= 200:
        list.append(int(list[i])+1) 
        return list[i]
    else:
        list.append(int(list[i])+0)
        return list[i]

def judgement(i):
    difference = abs(int(DIC['LEFTELBOW'][i])     - int(dic['LEFTELBOW'][i])) \
               + abs(int(DIC['RIGHTELBOW'][i])    - int(dic['RIGHTELBOW'][i])) \
               + abs(int(DIC['LEFTSHOULDER'][i])  - int(dic['LEFTSHOULDER'][i])) \
               + abs(int(DIC['RIGHTSHOULDER'][i]) - int(dic['RIGHTSHOULDER'][i])) \
               + abs(int(DIC['LEFTHIP'][i])       - int(dic['LEFTHIP'][i])) \
               + abs(int(DIC['RIGHTHIP'][i])      - int(dic['RIGHTHIP'][i])) \
               + abs(int(DIC['LEFTKNEE'][i])      - int(dic['LEFTKNEE'][i])) \
               + abs(int(DIC['RIGHTKNEE'][i])     - int(dic['RIGHTKNEE'][i]))

    if difference <= 100:
        return    'Marvelous'
          
                      
    elif difference <= 200:
        return    'Perfect'
          
    elif difference <= 300:
        return    'Good'
          
    else:
        return   'Bad'



# 內容為: 將照片預測節點後，算出8個角度
def Detect(frame):
    # Recolor image to RGB
    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.flags.writeable = False
    
    # Make detection
    Results = pose.process(Image)
  
    # Recolor back to BGR
    Image.flags.writeable = True
    Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
      
    # Extract landmarks
    try:          
        landmarks = Results.pose_landmarks.landmark
    except:
        pass
    try:    
        LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    except:
        pass
    try:
        LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    except:
        pass
    try:
        LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    except:
        pass
    try:
        LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    except:
        pass
    try:
        LEFT_KNEE=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    except:
        pass
    try:
        LEFT_ANKLE=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    except:
        pass
    try:
        RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    except:
        pass
    try:
        RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    except:
        pass
    try:
        RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    except:
        pass
    try:
        RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    except:
        pass
    try:
        RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    except:
        pass
    try:
        RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    except:
        pass


    # Calculate angle
    try:
        LEFTELBOW_angle = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    except:
        pass
    try:
        RIGHTELBOW_angle = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    except:
        pass
    try:
        LEFTSHOULDER_angle = calculate_angle(LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
    except:
        pass
    try:
        RIGHTSHOULDER_angle = calculate_angle(RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)
    except:
        pass
    try:
        LEFTHIP_angle = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    except:
        pass
    try:
        RIGHTHIP_angle = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    except:
        pass
    try:
        LEFTKNEE_angle = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    except:
        pass
    try:
        RIGHTKNEE_angle = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    except:
        pass
        
    try:
        DIC['LEFTELBOW'].append(LEFTELBOW_angle)
    except:
        DIC['LEFTELBOW'].append(0)

    try:
        DIC['RIGHTELBOW'].append(RIGHTELBOW_angle)
    except:
        DIC['RIGHTELBOW'].append(0)

    try:
        DIC['LEFTSHOULDER'].append(LEFTSHOULDER_angle)
    except:
        DIC['LEFTSHOULDER'].append(0)

    try:           
        DIC['RIGHTSHOULDER'].append(RIGHTSHOULDER_angle)
    except:
        DIC['RIGHTSHOULDER'].append(0)

    try:
        DIC['LEFTHIP'].append(LEFTHIP_angle)
    except:
        DIC['LEFTHIP'].append(0)

    try:
        DIC['RIGHTHIP'].append(RIGHTHIP_angle)
    except:
        DIC['RIGHTHIP'].append(0)

    try:
        DIC['LEFTKNEE'].append(LEFTKNEE_angle)
    except:
        DIC['LEFTKNEE'].append(0)

    try:
        DIC['RIGHTKNEE'].append(RIGHTKNEE_angle)
    except:
        DIC['RIGHTKNEE'].append(0)
    
    
    
    i = len(DIC['LEFTELBOW'])-1

    LEFTELBOW =  abs(int(DIC['LEFTELBOW'][i]) - int(dic['LEFTELBOW'][i]))
    RIGHTELBOW =  abs(int(DIC['RIGHTELBOW'][i]) - int(dic['RIGHTELBOW'][i]))
    LEFTSHOULDER =  abs(int(DIC['LEFTSHOULDER'][i]) - int(dic['LEFTSHOULDER'][i]))
    RIGHTSHOULDER =  abs(int(DIC['RIGHTSHOULDER'][i]) - int(dic['RIGHTSHOULDER'][i]))
    LEFTHIP =  abs(int(DIC['LEFTHIP'][i]) - int(dic['LEFTHIP'][i]))
    RIGHTHIP =  abs(int(DIC['RIGHTHIP'][i]) - int(dic['RIGHTHIP'][i]))
    LEFTKNEE =  abs(int(DIC['LEFTKNEE'][i]) - int(dic['LEFTKNEE'][i]))
    RIGHTKNEE =  abs(int(DIC['RIGHTKNEE'][i]) - int(dic['RIGHTKNEE'][i]))

    total_score = score(i)
          



    # 印出骨架
    mp_drawing.draw_landmarks(
        Image,
        Results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      
    # Render detections 將各點角度值畫上圖  語法: cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    try:
        cv2.putText(Image, str(LEFTELBOW), 
                    tuple(np.multiply(LEFT_ELBOW, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(RIGHTELBOW), 
                    tuple(np.multiply(RIGHT_ELBOW, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(LEFTSHOULDER), 
                    tuple(np.multiply(LEFT_SHOULDER, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(RIGHTSHOULDER), 
                    tuple(np.multiply(RIGHT_SHOULDER, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(LEFTHIP), 
                    tuple(np.multiply(LEFT_HIP, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(RIGHTHIP), 
                    tuple(np.multiply(RIGHT_HIP, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(LEFTKNEE), 
                    tuple(np.multiply(LEFT_KNEE, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Image, str(RIGHTKNEE), 
                    tuple(np.multiply(RIGHT_KNEE, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 在圖上加方框  語法: cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
        # 分數的底
        cv2.rectangle(Image, (0,0), (180,50), (255,255,255), -1)
        cv2.rectangle(Image, (0,380), (150,480), (200,200,200), -1)

        # 在視訊鏡頭上加上累計分數及當下的評分表現
        cv2.putText(Image, 'SCORE', 
                (20,420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        cv2.putText(Image, str(judgement(i)), 
                (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
        cv2.putText(Image, str(total_score), 
                (30,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', Image)
    except:
        cv2.imshow('Mediapipe Feed', Image)



with open(path + '.json', 'r') as f:
    dic = json.load(f)



    switch = False
pygame.init()
pygame.mixer.music.load(path + '.mp3')


Cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('detect_' + path + '.mp4')
t = time.time()
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_num = 0
    while Cap.isOpened() or cap.isopened():
        if cap.get(1) / cap.get(5) > time.time()-t:
            time.sleep(0.04)

        if frame_num == len(dic['LEFTELBOW']):
            break
        retval, frame = Cap.read()
        retval, frame1 = cap.read()

        if retval == True:
            frame = cv2.flip(frame, 1)
            Detect(frame)
            cv2.imshow('Video',frame1)
            if switch == False:
                switch = True
                pygame.mixer.music.play(0)

            # try:
            #     Detect(frame)
            
            #     cv2.imshow('Video',frame1)
            # except:
            #     continue
            frame_num += 1
        else:
            cap.release()
            break
          
        
        if cv2.waitKey(4) & 0xFF == 27:
            
            break
pygame.mixer.music.stop()
cv2.destroyWindow('Video')
Cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()