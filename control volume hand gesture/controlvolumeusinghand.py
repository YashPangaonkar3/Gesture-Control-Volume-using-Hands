# 1st install library openCv-contrib
# 2nd install a model / library mediapipe
# for volume control requried two model that is pycaw-rf [Python Core Audio Windows Library] and pycaw 

import cv2
import mediapipe as mp
from numpy import interp
from math import sqrt,pow
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
from ctypes import cast,POINTER


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands
minvalue = 30
maxvalue = 0

handmodel = mp_hand.Hands(
    model_complexity = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,CLSCTX_ALL,None
)
volume = cast(interface,POINTER(IAudioEndpointVolume))
minvolume,maxvolume,_ = volume.GetVolumeRange()
def detect_hand(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = handmodel.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image,results):
    global maxvalue,minvalue,maxvolume,minvolume
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            l = hand_landmarks.landmark
            h,w,_ = image.shape
            thumb_tip = (round(l[4].x*w),round(l[4].y*h))
            indexfinger_tip = (round(l[8].x*w),round(l[8].y*h))
            thumb2 = (round(l[2].x*w),round(l[2].y*h))
            indexfinger5 = (round(l[5].x*w),round(l[5].y*h))
            distance = sqrt(pow(thumb_tip[0]-indexfinger_tip[0],2)+pow(thumb_tip[1]-indexfinger_tip[1],2))
            distance2 = sqrt(pow(thumb2[0]-indexfinger5[0],2)+pow(thumb2[1]-indexfinger5[1],2))
            distance,distance2 = round(distance),round(distance2)
            if maxvalue == 0:
                maxvalue = distance2*2
            if abs(distance-2*distance2) < 5:
                maxvalue = distance2*2
            volumetoset = interp(distance,[minvalue,maxvalue],[minvolume,maxvolume])
            volume.SetMasterVolumeLevel(volumetoset,None)
            cv2.putText(image,str(distance),(0,20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.putText(image,str(distance2*2),(0,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.line(image,indexfinger_tip,thumb_tip,(255,0,0),2)
            cv2.circle(image,thumb_tip,15,(255,0,0),-1)
            cv2.circle(image,indexfinger_tip,15,(255,0,0),-1)
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    _,frame = webcam.read()
    frame,results = detect_hand(frame)
    draw_landmarks(frame,results)
    cv2.imshow("output",frame)
    cv2.waitKey(27)