import cv2, json, pygame, time
import mediapipe as mp
import numpy as np

path = 'demo4'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

points = ['LEFTELBOW', 'RIGHTELBOW', 'LEFTSHOULDER', 'RIGHTSHOULDER', 'LEFTHIP', 'RIGHTHIP', 'LEFTKNEE', 'RIGHTKNEE']
count_score = [0,0,0,0,0]   # excellent, great, good, bad, terrible

dic = dict()
DIC = dict()
for x in points:
    dic[x] = []

# 分數的list
# list=[0]
total_score = 0
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



def score(i):
    # 各部位角度差之平均
    Difference = 0
    count = 0
    anser = 0
    for x in points:
        try:
            Difference += abs(int(DIC[x]) - int(dic[x][i]))
            count += 1
        except:
            pass
    if count == 0:
        return anser
    mean_angle = Difference/count
    if mean_angle <= 10:
        anser = 5
    elif mean_angle <= 20:
        anser = 4
    elif mean_angle <= 30:
        anser = 3
    elif mean_angle <= 40:
        anser = 2
    elif mean_angle <= 50:
        anser = 1
    return anser


def judgement(i):
    Difference = 0
    count = 0
    for x in points:
        try:
            Difference += abs(int(DIC[x]) - int(dic[x][i]))
            count += 1
        except:
            pass
    if count == 0:
        return 'Unknown'
    mean_angle = Difference/count
    if mean_angle <= 10:
        count_score[0] += 1
        return 'Excellent'
    elif mean_angle <= 20:
        count_score[1] += 1
        return 'Great'
    elif mean_angle <= 30:
        count_score[2] += 1
        return 'Good'
    elif mean_angle <= 50:
        count_score[3] += 1
        return 'Bad'
    else:
        count_score[4] += 1
        return 'Terrible'


# 內容為: 將照片預測節點後，算出8個角度
def Detect(frame):
    global total_score
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
        DIC['LEFTELBOW'] = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    except:
        DIC['LEFTELBOW'] = None
    try:
        DIC['RIGHTELBOW'] = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    except:
        DIC['RIGHTELBOW'] = None
    try:
        DIC['LEFTSHOULDER'] = calculate_angle(LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
    except:
        DIC['LEFTSHOULDER'] = None
    try:
        DIC['RIGHTSHOULDER'] = calculate_angle(RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)
    except:
        DIC['RIGHTSHOULDER'] = None
    try:
        DIC['LEFTHIP'] = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    except:
        DIC['LEFTHIP'] = None
    try:
        DIC['RIGHTHIP'] = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    except:
        DIC['RIGHTHIP'] = None
    try:
        DIC['LEFTKNEE'] = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    except:
        DIC['LEFTKNEE'] = None
    try:
        DIC['RIGHTKNEE'] = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    except:
        DIC['RIGHTKNEE'] = None
        
    
    
    
    
    # i = len(DIC['LEFTELBOW'])-1
    try:
        LEFTELBOW =  abs(int(DIC['LEFTELBOW']) - int(dic['LEFTELBOW'][i]))
    except:
        LEFTELBOW = None
    try:
        RIGHTELBOW =  abs(int(DIC['RIGHTELBOW']) - int(dic['RIGHTELBOW'][i]))
    except:
        RIGHTELBOW = None
    try:
        LEFTSHOULDER =  abs(int(DIC['LEFTSHOULDER']) - int(dic['LEFTSHOULDER'][i]))
    except:
        LEFTSHOULDER = None
    try:
        RIGHTSHOULDER =  abs(int(DIC['RIGHTSHOULDER']) - int(dic['RIGHTSHOULDER'][i]))
    except:
        RIGHTSHOULDER = None
    try:
        LEFTHIP =  abs(int(DIC['LEFTHIP']) - int(dic['LEFTHIP'][i]))
    except:
        LEFTHIP = None
    try:
        RIGHTHIP =  abs(int(DIC['RIGHTHIP']) - int(dic['RIGHTHIP'][i]))
    except:
        RIGHTHIP = None
    try:
        LEFTKNEE =  abs(int(DIC['LEFTKNEE']) - int(dic['LEFTKNEE'][i]))
    except:
        LEFTKNEE = None
    try:
        RIGHTKNEE =  abs(int(DIC['RIGHTKNEE']) - int(dic['RIGHTKNEE'][i]))
    except:
        RIGHTKNEE = None


    total_score += score(i)
          



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

        # for c, x in enumerate(points):
        #     if DIC[x] == None:
        #         ll = 'None'
        #     else:
        #         ll = str(int(DIC[x]))
        #     cv2.putText(Image, ll, (20, 80+c*40), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # for c, x in enumerate(points):
        #     if dic[x][i] == None:
        #         ll = 'None'
        #     else:
        #         ll = str(int(dic[x][i]))
        #     cv2.putText(Image, ll, (90, 80+c*40), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.imshow('Webcam', Image)
    except:
        # for c, x in enumerate(points):
        #     if DIC[x] == None:
        #         ll = 'None'
        #     else:
        #         ll = str(int(DIC[x]))
        #     cv2.putText(Image, ll, (20, 80+c*40), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # for c, x in enumerate(points):
        #     if dic[x][i] == None:
        #         ll = 'None'
        #     else:
        #         ll = str(int(dic[x][i]))
        #     cv2.putText(Image, ll, (90, 80+c*40), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Webcam', Image)



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
        if cap.get(1) / cap.get(5) < time.time()-t-0.04:
            i += 1
            cap.set(1, cap.get(1)+1)
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
# cv2.waitKey(0)
cv2.destroyAllWindows()
end_img = cv2.imread('score.jpg')
totoal_count_score = sum(count_score)
cv2.putText(end_img, str(total_score), (404, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(end_img, str(int(100*count_score[0]/totoal_count_score)) + ' %', (404, 284), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(end_img, str(int(100*count_score[1]/totoal_count_score)) + ' %', (404, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(end_img, str(int(100*count_score[2]/totoal_count_score)) + ' %', (404, 377), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(end_img, str(int(100*count_score[3]/totoal_count_score)) + ' %', (404, 419), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(end_img, str(int(100*count_score[4]/totoal_count_score)) + ' %', (404, 462), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imshow('summarize', end_img)
cv2.waitKey(0)
